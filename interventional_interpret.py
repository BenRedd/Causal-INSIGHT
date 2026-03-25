import torch
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.model as module_arch
import argparse
import torch.nn.functional as F
from evaluation.evaluator import evaluate, getextendeddelays, evaluatedelay
from parse_config import ConfigParser
from utils import prepare_device
from sklearn.metrics import precision_score, recall_score, f1_score

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# For running models directly
dataset_number = 4

def read_groundtruth_csv(file_path: str):
    """
    Reads a ground-truth CSV file into a list of tuples.
    Supports rows with 3 or 4+ columns (e.g., (i,j,t) or (i,j,t,value)).
    """
    df = pd.read_csv(file_path, header=None)
    return [tuple(x) for x in df.to_records(index=False)]


def groundtruth_edge_set(gt_tuples):
    """Return set of directed edges (i,j) from GT tuples (i,j,t[,val])."""
    edges = set()
    for tup in gt_tuples:
        i, j = int(tup[0]), int(tup[1])
        edges.add((i, j))
    return edges


def predicted_edge_set(predicted_relationships):
    """
    predicted_relationships is dict: child_j -> [parent_i, ...]
    Convert to directed edge set: parent_i -> child_j
    """
    edges = set()
    for j, parents in predicted_relationships.items():
        for i in parents:
            edges.add((int(i), int(j)))
    return edges


def compute_struct_metrics(gt_edges, pred_edges, N: int, include_self_loops: bool = False):
    """
    Structure-only metrics:
      - SHD (here: FP + FN; direction-aware)
      - TPR (Recall): TP/(TP+FN)
      - FDR: FP/(TP+FP)
    """
    # Universe of possible directed edges
    total_possible = 0
    for i in range(N):
        for j in range(N):
            if not include_self_loops and i == j:
                continue
            total_possible += 1

    gt = set(gt_edges)
    pred = set(pred_edges)

    TP = sum(1 for e in pred if e in gt)
    FP = sum(1 for e in pred if e not in gt)
    FN = sum(1 for e in gt if e not in pred)
    TN = total_possible - TP - FP - FN

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Direction-aware SHD approximation (adds/removals only)
    SHD = FP + FN

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "TPR": TPR, "FDR": FDR, "SHD": SHD}


def compute_precision_of_delay(groundtruth_tuples, predicted_tuples):
    """
    Kept for compatibility, but your main pipeline uses evaluatedelay().
    This assumes tuples are (i,j,t) and compares exact lag matches.
    """
    groundtruth_edges = {(i, j): t for i, j, t in groundtruth_tuples}
    predicted_edges = {(i, j): t for i, j, t in predicted_tuples}

    true_positives = 0
    correct_time_steps = 0

    for (i, j), predicted_t in predicted_edges.items():
        if (i, j) in groundtruth_edges:
            true_positives += 1
            if predicted_t == groundtruth_edges[(i, j)]:
                correct_time_steps += 1

    if true_positives == 0:
        return 0.0

    return correct_time_steps / true_positives


def extract_causal_relationships(S_retained, T_masked, time_base=1):
    """
    Extracts (i, j, t) relationships from pruned causal tensor slices.

    Args:
        S_retained (torch.Tensor or np.ndarray): (N, N) retained signal strengths.
        T_masked (torch.Tensor or np.ndarray): (N, N) time indices (masked elsewhere).
        time_base (int): Add 1 to make time index 1-based if needed.

    Returns:
        List of tuples: [(i, j, t), ...]
    """
    N, _ = S_retained.shape
    relationships = []

    for i in range(N):
        for j in range(N):
            if float(S_retained[i, j]) > 0:
                t = int(T_masked[i, j])
                if t >= 0:
                    relationships.append((i, j, int(t) + time_base))

    return relationships


def compute_edge_metrics(groundtruth_tuples, predicted_tuples):
    # Extract unique (i, j) pairs from tuples
    groundtruth_edges = set((i, j) for i, j, t in groundtruth_tuples)
    predicted_edges = set((i, j) for i, j, t in predicted_tuples)

    # Create the universe of all considered edges
    all_edges = groundtruth_edges.union(predicted_edges)

    y_true = [1 if edge in groundtruth_edges else 0 for edge in all_edges]
    y_pred = [1 if edge in predicted_edges else 0 for edge in all_edges]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def load_model(path, args, name=f'Batch Runner/0430_111733/fMRI{dataset_number}', run_id=None):
    config_path = path + '/config.json'
    checkpoint_path = path + '/model_best.pth'
    args_dict = {
        'name': name,
        'config': config_path,
        'resume': None,
        'device': args.device
    }
    config = ConfigParser.from_args(args=args_dict, run_id=run_id)

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    config['data_loader']['args']['series_num'] = data_loader.series_num
    config['data_loader']['args']['time_step'] = data_loader.time_step
    config['data_loader']['args']['output_window'] = data_loader.output_window

    # build model architecture
    model = config.init_obj('arch', module_arch, config)

    # prepare device(s)
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    if len(device_ids) > 1 and device.type != "mps":
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if device.type == "mps":
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('mps'), weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(checkpoint['state_dict'])
    return model, config, data_loader


def eval(logger, gt, allcauses, alldelays, columns):
    FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(
        logger, gt, allcauses, columns
    )
    extendeddelays, readgt, extendedreadgt = getextendeddelays(gt, columns)
    percentagecorrect = evaluatedelay(extendeddelays, alldelays, TPs, 1) * 100
    logger.info(f"Percentage of delays that are correctly discovered: {percentagecorrect}%")
    return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct, percentagecorrect


def find_first_stable_minimum(Q, m_hat, window=5, delta=1.5):
    Q = np.array(Q)
    m_hat = np.array(m_hat)
    n = len(Q)

    for i in range(n - window):
        if i > 0 and Q[i] > Q[i - 1]:
            continue
        if i < n - 1 and Q[i] > Q[i + 1]:
            continue

        next_vals = Q[i + 1: i + 1 + window]
        net_deviation = np.mean(np.abs(next_vals - Q[i]))

        if net_deviation <= delta:
            return int(m_hat[i]), Q[i], i

    i = np.argmin(Q)
    return int(m_hat[i]), Q[i], i


def main(model, args, config, data_loader, gt):
    logger = config.get_logger('train')
    logger.info("===================Running Interventional Interpretation===================")
    logger.info("ground_truth:" + (gt if gt else "None"))

    device, device_ids = prepare_device(config['n_gpu'])

    columns = list(data_loader.df_data.columns)
    rows = len(data_loader.df_data)
    window_size = data_loader.time_step

    # intervention setting
    intervention_val = 0.0 if (gt and ('sim' in gt or 'fMRI' in gt)) else 1.0

    model = model.to(device)
    model.eval()
    S_list = []

    # ----- probing / intervention stage -----
    for batch_data, _ in data_loader:
        with torch.no_grad():
            X = batch_data.to(device)          # [B, N, W]
            X_hat = model(X)                   # [B, N, W_out]
            B, N, _ = X.shape

            # Vectorized intervention
            X_int = X.repeat(N, 1, 1)          # [N*B, N, W]
            eye = torch.eye(N, device=device).unsqueeze(1).repeat(1, B, 1).reshape(N * B, N)

            # clamp time index 0 for each variable i across B
            X_int[torch.arange(N * B), :, 0] = torch.where(
                eye.bool(),
                torch.tensor(intervention_val, device=device),
                X_int[torch.arange(N * B), :, 0]
            )

            X_hat_prime = model(X_int).reshape(N, B, N, -1).permute(1, 0, 2, 3)  # [B, N, N, W_out]
            diffs = X_hat_prime - X_hat.unsqueeze(1)                              # [B, N, N, W_out]
            avg_diffs = diffs.mean(dim=0)                                         # [N, N, W_out]

            S_list.append(avg_diffs.abs())

    S = torch.stack(S_list).mean(dim=0)  # [N, N, W_out]
    S_peak, T_peak = S.max(dim=2)
    # N is defined in loop above; keep consistent
    N = S_peak.shape[0]

    # ----- Qbic sweep stage -----
    results = []
    patience = 15
    best_Qbic = float('inf')
    no_improve_counter = 0

    for m_hat in range(1, (N ** 2 + 1)):
        flat = S_peak.flatten()
        topk_values, _ = torch.topk(flat, m_hat, largest=True)
        threshold = topk_values.min()

        mask = S_peak >= threshold
        S_retained = torch.where(mask, S_peak, torch.tensor(0.0, device=device))

        # prune bidirectional by keeping the stronger direction
        for i in range(N):
            for j in range(i + 1, N):
                if S_retained[i, j] > S_retained[j, i]:
                    S_retained[j, i] = 0.0
                else:
                    S_retained[i, j] = 0.0

        T_masked = torch.where(mask, T_peak, torch.tensor(0.0, device=device))
        relationships = extract_causal_relationships(S_retained, T_masked)

        predicted_relationships = {i: [] for i in range(N)}
        predicted_delays = {}
        for i, j, delay in relationships:
            predicted_relationships[j].append(i)      # parent i -> child j
            predicted_delays[(j, i)] = delay          # evaluator expects (child, parent)

        n = rows - window_size + 1
        mse_per_variable = [0.0 for _ in range(N)]

        for batch_data, batch_labels in data_loader:
            X = batch_data.to(device)
            Y = batch_labels.to(device)

            for j in range(N):
                parents = predicted_relationships[j]
                X_masked = X.clone()

                for var in range(N):
                    if var not in parents:
                        X_masked[:, var, :] = 0.0

                with torch.no_grad():
                    Y_hat = model(X_masked)

                Y_hat_j = Y_hat[:, j, :]
                Y_j = Y[:, j, :]
                mse = F.mse_loss(Y_hat_j, Y_j, reduction='sum').item()
                mse_per_variable[j] += mse

        Qbic = sum([
            n * np.log((mse_per_variable[j] / n) + 1e-12) + 0.4 * len(predicted_relationships[j]) * np.log(n)
            for j in range(N)
        ])

        results.append({
            "m_hat": m_hat,
            "Qbic": Qbic,
            "relationships": predicted_relationships,
            "delays": predicted_delays,
        })

        if Qbic < best_Qbic:
            best_Qbic = Qbic
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if no_improve_counter >= patience:
            print(f"\nEarly stopping at m̂ = {m_hat} (Qbic stopped improving for {patience} steps)")
            break

    min_m_hat, min_Qbic, best_index = find_first_stable_minimum(
        [r["Qbic"] for r in results], [r["m_hat"] for r in results]
    )
    best_result = results[best_index]

    # ----- evaluate (existing evaluator provides F1 and PoD) -----
    FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct, perc_correct = eval(
        logger, gt, best_result["relationships"], best_result["delays"], columns
    )

    precision = TPdirect / (TPdirect + FPdirect) if (TPdirect + FPdirect) > 0 else 0.0
    recall = TPdirect / (TPdirect + FN) if (TPdirect + FN) > 0 else 0.0

    # ----- structure-only metrics (SHD/TPR/FDR) -----
    struct_metrics = {"SHD": None, "TPR": None, "FDR": None}
    if gt:
        gt_tuples = read_groundtruth_csv(gt)
        gt_edges = groundtruth_edge_set(gt_tuples)
        pred_edges = predicted_edge_set(best_result["relationships"])
        struct_metrics = compute_struct_metrics(gt_edges, pred_edges, N, include_self_loops=False)

        logger.info(
            f"Structure metrics: SHD={struct_metrics['SHD']}  "
            f"TPR={struct_metrics['TPR']:.4f}  FDR={struct_metrics['FDR']:.4f}"
        )

    logger.info(f"Optimal m̂ = {min_m_hat} with Qbic = {min_Qbic}")
    logger.info(f"Optimal F1 = {F1direct:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": float(F1direct),
        "PoD": float(perc_correct) / 100.0,
        "SHD": struct_metrics["SHD"],
        "TPR": struct_metrics["TPR"],
        "FDR": struct_metrics["FDR"],
    }


"""
Run causal inference directly
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-d', '--device', default="0", type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-m', '--model_type', default="cnn_lstm", type=str,
                        help='Specify model type: mlp, lstm, prob')
    args = parser.parse_args()

    def render(_args):
        # Adjust these paths per dataset/model as needed
        model_tuple = load_model(
            'saved/models/Batch Runner/0720_221713/lorenz0/model',
            _args
        )
        gt_path = 'data/lorenz/groundtruth.csv'
        bigdata = False
        return model_tuple, gt_path, bigdata

    (model, config, data_loader), gt, bigdata = render(args)