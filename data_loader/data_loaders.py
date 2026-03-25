from sklearn import preprocessing
import pandas as pd
from base import BaseDataLoader

class TimeseriesDataLoader(BaseDataLoader):
    """
    DataLoader for time series causal discovery.
    """
    def __init__(self, data_dir, batch_size, time_step, feature_dim, output_dim, output_window,
                 shuffle=True, validation_split=0.0, num_workers=1):
        self.df = pd.read_csv(data_dir)
        self.data = self.df.values.astype('float32')  # Convert to NumPy array

        # Standardize for stability
        scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
        self.data = scaler.fit_transform(self.data)

        self.time_step = time_step
        self.series_num = self.data.shape[1]  # Number of variables (N)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.output_window = output_window

        # Create sliding windows
        self.data_dir = data_dir
        self.df_data = pd.read_csv(self.data_dir)
        self.data_len = len(self.df_data.index)
        self.data = self.df_data.values.astype('float32')

        # Construct input samples
        # Sanity check
        assert self.time_step < len(self.data) + 1, "time_steps must be shorter than dataset length."
        assert self.output_window < self.time_step, "output_window must be shorter than time_steps."

        # Create dataset
        self.dataset = [
            (
                self.data[i - self.time_step:i].T,  # Shape: (num_series, time_steps)
                self.data[i - self.output_window:i].T  # Shape: (num_series, output_window)
            )
            for i in range(self.time_step, len(self.data) + 1)
        ]

        # Create dataset with sliding windows
        self.dataset = self._construct_dataset()

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _construct_dataset(self):
        dataset = []
        for i in range(self.time_step, len(self.data) + 1):
            X = self.data[i - self.time_step:i].T  # Shape: (N, T)
            Y = self.data[i - self.output_window:i].T  # Shape: (N, output_window)
            dataset.append((X, Y))
        return dataset
