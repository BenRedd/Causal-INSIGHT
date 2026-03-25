import argparse
import interventional_interpret
import pandas as pd
from pathlib import Path
from utils import read_json, prepare_device
from parse_config import ConfigParser
from datetime import datetime
import train
import torch
import numpy as np
import time

def construct_demo():
    task_list = {}
    for i in [8]:
        task_list[f'fMRI{i}'] = {
            'dataset': f"data/fMRI/timeseries{i}.csv",
            'groundtruth': f"data/fMRI/sim{i}_gt_processed.csv"
        }
    return task_list

def construct_basic_v():
    task_list = {}
    for i in range(10):
        task_list[f'v{i}'] = {
            'dataset': f"data/basic/v/data_{i}.csv",
            'groundtruth': f"data/basic/v/groundtruth.csv"
        }
    return task_list

def construct_basic_diamond():
    task_list = {}
    for i in range(10):
        task_list[f'diamond{i}'] = {
            'dataset': f"data/basic/diamond/data_{i}.csv",
            'groundtruth': f"data/basic/diamond/groundtruth.csv"
        }
    return task_list

def construct_basic_fork():
    task_list = {}
    for i in range(10):
        task_list[f'fork{i}'] = {
            'dataset': f"data/basic/fork/data_{i}.csv",
            'groundtruth': f"data/basic/fork/groundtruth.csv"
        }
    return task_list

def construct_basic_mediator():
    task_list = {}
    for i in range(10):
        task_list[f'mediator{i}'] = {
            'dataset': f"data/basic/mediator/data_{i}.csv",
            'groundtruth': f"data/basic/mediator/groundtruth.csv"
        }
    return task_list

def construct_fMRI():
    task_list = {}
    for i in range(1,29):
        task_list[f'fMRI{i}'] = {
            'dataset': f"data/fMRI/timeseries{i}.csv",
            'groundtruth': f"data/fMRI/sim{i}_gt_processed.csv"
        }
    return task_list

def construct_lorenz():
    task_list = {}
    for i in range(10):
        task_list[f'lorenz{i}'] = {
            'dataset': f"data/lorenz/timeseries{i}.csv",
            'groundtruth': f"data/lorenz/groundtruth.csv"
        }
    return task_list

tasks = {
    'v': construct_basic_v,
    'fork': construct_basic_fork,
    'diamond': construct_basic_diamond,
    'mediator': construct_basic_mediator,
    'fMRI': construct_fMRI,
    "lorenz": construct_lorenz,
    "demo": construct_demo,
}

def runtask(label, args, dataset, ground_truth, task_name):
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    args_dict = {
        'name': f'Batch Runner/{label}/{task_name}',
        'config': args.config,
        'resume': None,
        'device': args.device,
        'data_dir': dataset
    }

    config = ConfigParser.from_args(args=args_dict, run_id='model')

    # -------- TRAIN --------
    train.main(config)

    # -------- LOAD MODEL --------
    model, config, data_loader = interventional_interpret.load_model(
        f'saved/models/Batch Runner/{label}/{task_name}/model',
        args,
        f'Batch Runner/{label}/{task_name}',
        'inference'
    )

    # -------- TIME INTERVENTIONAL INTERPRETATION --------
    device, _ = prepare_device(config['n_gpu'])
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    metrics = interventional_interpret.main(
        model, args, config, data_loader, ground_truth
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    wallclock = end_time - start_time
    metrics["WallClockSec"] = wallclock

    # -------- PRINT PER-DATASET RUNTIME --------
    print(f"[{task_name}] WallClock(s): {wallclock:.2f}")

    return metrics

def main(args):
    task_list = tasks[args.task]()
    label = datetime.now().strftime(r'%m%d_%H%M%S')

    all_results = []

    for task_name, task_msg in task_list.items():
        result = runtask(label, args, task_msg['dataset'], task_msg['groundtruth'], task_name)
        result["Task"] = task_name  # track which task each result is for
        all_results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save summary CSV
    configJSON = read_json(args.config)
    save_dir = Path(configJSON['trainer']['save_dir'])
    summary_dir = save_dir / 'log' / 'Batch Runner' / label / 'summary.csv'
    summary_dir.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir, index=False)
    cols = ['Task'] + [col for col in df.columns if col != 'Task']
    df = df[cols]

    # Print summary
    print("\n=================== Summary ===================")
    print('\t' + df.to_string(index=False).replace('\n', '\n\t'))

    # Compute mean and std (excluding 'Task')
    metrics = df.drop(columns=["Task"])
    means = metrics.mean()
    stds = metrics.std()

    # Combine into one summary table
    summary_stats = pd.DataFrame({
        'Mean': means,
        'Std': stds
    })

    # Print with aligned formatting
    print("\n================== Averages ===================")
    print('\t' + summary_stats.to_string(formatters={
        'Mean': '{:.4f}'.format,
        'Std': '{:.4f}'.format
    }).replace('\n', '\n\t'))

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Inference')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--task', default='fMRI', type=str,
                      help='task (default: fMRI)')
    args = args.parse_args()

    label = main(args)
