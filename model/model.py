import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# -----------------------------
# Causal masking front-end
# -----------------------------
class CausalConv1D(BaseModel):
    """
    Masked conv front-end that enforces:
      - access to all variables at lags 1..K-1
      - access to other variables at lag 0
      - NO access to self at lag 0 (prevents leakage/self-copy)
    Input:  (B, N, T)
    Output: (B, H, T)
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.K = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=True,
        )

        # Mask: allow all inputs for lags 1..K-1
        mask = torch.zeros_like(self.conv.weight.data)
        mask[:, :, :kernel_size - 1] = 1.0

        # For lag 0 (last index), block "self"
        # We interpret output channels as grouped by variable index modulo N
        for out_ch in range(self.out_channels):
            out_var = out_ch % self.in_channels
            for in_ch in range(self.in_channels):
                if in_ch != out_var:
                    mask[out_ch, in_ch, kernel_size - 1] = 1.0

        self.register_buffer("mask", mask)

    def forward(self, x):
        x_padded = F.pad(x, (self.K - 1, 0))  # left pad only
        w = self.conv.weight * self.mask
        y = F.conv1d(x_padded, w, bias=self.conv.bias, stride=1)
        return y  # (B, H, T)


# -----------------------------
# Causal temporal CNN head
# -----------------------------
class CausalTemporalCNNHead(BaseModel):
    """
    A true CNN predictor head operating over time dimension T.
    Uses left-padding only to preserve causality (no future leakage).
    Input:  (B, T, H)
    Output: (B, T, N)
    """
    def __init__(self, hidden_dim, num_series, head_channels=None, head_kernel_size=3, dropout=0.0):
        super().__init__()
        if head_channels is None:
            head_channels = hidden_dim

        self.hidden_dim = hidden_dim
        self.num_series = num_series
        self.k = head_kernel_size

        # conv over time: (B, H, T) -> (B, C, T)
        self.conv1 = nn.Conv1d(hidden_dim, head_channels, kernel_size=self.k, padding=0, bias=True)
        self.conv2 = nn.Conv1d(head_channels, head_channels, kernel_size=self.k, padding=0, bias=True)

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.proj = nn.Linear(head_channels, num_series)

    def forward(self, x):
        # x: (B, T, H)
        x = x.permute(0, 2, 1)  # (B, H, T)

        # causal padding (left only)
        x = F.pad(x, (self.k - 1, 0))
        x = self.act(self.conv1(x))
        x = self.drop(x)

        x = F.pad(x, (self.k - 1, 0))
        x = self.act(self.conv2(x))
        x = self.drop(x)

        x = x.permute(0, 2, 1)  # (B, T, C)
        return self.proj(x)     # (B, T, N)


# -----------------------------
# Main model
# -----------------------------
class CausalInsight(BaseModel):
    def __init__(self, config, model_type, hidden_dim, lstm_hidden_dim, kernel_size,
                 mlp_hidden_dim=None, cnn_head_channels=None, cnn_head_kernel_size=3, dropout=0.0):
        super().__init__()

        self.model_type = model_type
        self.num_series = config["data_loader"]["args"].get("series_num")
        self.output_window = config["data_loader"]["args"].get("output_window")
        self.hidden_dim = hidden_dim

        # Shared causal masking front-end
        self.causal_conv = CausalConv1D(
            in_channels=self.num_series,
            out_channels=self.hidden_dim,
            kernel_size=kernel_size
        )

        # Heads
        if model_type == "linear":
            self.act = nn.LeakyReLU()
            self.head = nn.Linear(hidden_dim, self.num_series)

        elif model_type == "mlp":
            if mlp_hidden_dim is None:
                mlp_hidden_dim = hidden_dim
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mlp_hidden_dim, self.num_series),
            )

        elif model_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Linear(lstm_hidden_dim, self.num_series)

        elif model_type == "cnn":
            self.head = CausalTemporalCNNHead(
                hidden_dim=hidden_dim,
                num_series=self.num_series,
                head_channels=cnn_head_channels,
                head_kernel_size=cnn_head_kernel_size,
                dropout=dropout,
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, X):
        # X: (B, N, T)
        x = self.causal_conv(X)       # (B, H, T)
        x = x.permute(0, 2, 1)        # (B, T, H)

        if self.model_type == "linear":
            x = self.act(x)
            x_out = self.head(x)      # (B, T, N)

        elif self.model_type == "mlp":
            x_out = self.head(x)      # (B, T, N)

        elif self.model_type == "lstm":
            lstm_out, _ = self.lstm(x)  # (B, T, lstm_hidden_dim)
            x_out = self.head(lstm_out) # (B, T, N)

        elif self.model_type == "cnn":
            x_out = self.head(x)      # (B, T, N)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # keep only the last output_window predictions
        x_out = x_out[:, -self.output_window:, :]  # (B, out_w, N)
        x_out = x_out.permute(0, 2, 1)             # (B, N, out_w)
        return x_out
