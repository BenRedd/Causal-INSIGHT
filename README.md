# Causal-INSIGHT

---

## 📄 Paper

- arXiv: https://arxiv.org/abs/2603.25473
- Accepted at **IEEE International Joint Conference on Neural Networks (IJCNN), 2026**

---

## 🚀 Quick Start

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Run the demo (example fMRI dataset):

```bash
python runner.py -c config/config_fMRI.json -t demo
```

---

## 🧠 Overview

Causal-INSIGHT is a **model-agnostic, post-hoc interpretability framework** for extracting directed, time-lagged influence structure from trained temporal predictors.

The method operates by:
- probing trained models via **intervention-inspired input clamping**
- constructing **temporal influence signals**
- selecting sparse causal graphs using the **Qbic** criterion

## 🧩 Supported Architectures

This repository implements Causal-INSIGHT with multiple backbone predictors.  
All models share a common **causal masking front-end**, which prevents temporal leakage by restricting access to future information and disallowing self-access at the current timestep.

The masking mechanism is identical across architectures, allowing fair comparison while isolating the effect of different temporal inductive biases.

The following backbone families are supported:

### 🔹 MLP
A multilayer perceptron serving as a minimal baseline for modeling temporal dependencies from the masked input representation.

### 🔹 CNN
A temporal convolutional network that captures short-range temporal patterns using convolutional filters.

### 🔹 LSTM
A recurrent model designed to capture longer-range dependencies via gated recurrence.

### 🔹 CausalFormer (External)
A transformer-based temporal model originally proposed for causal discovery.  
In this repository, it is treated as a black-box predictor for comparison purposes (see instructions below for integration).

---

## 🔌 Using External Model: CausalFormer

CausalFormer is not included in this repository.

To reproduce results:

1. Train a model using the official implementation:  
   https://github.com/lingbai-kong/CausalFormer  

2. Save the trained model weights (e.g. `model_best.pth`)  

3. Adapt the inference pipeline (`interventional_interpret.py`) to load the trained model (expected dimensions may differ)  

4. Run the Causal-INSIGHT probing method on the trained model  

---

## ⚙️ Requirements

- Python 3.10  
- PyTorch 2.1.0  
- CUDA 11.8 *(optional for GPU acceleration)*  

### Required Packages

- torchvision  
- numpy  
- pandas  
- scikit-learn  
- tensorboardx  

---

## 📁 Repository Structure

```
Causal-INSIGHT/
│
├── runner.py                         # Main entry point (train + inference)
├── train.py                          # Training script
├── interventional_interpret.py       # Core probing / inference method
├── parse_config.py                   # Config parser and CLI handling
│
├── config/                           # Configuration files for experiments
├── data/                             # Dataset storage (user-provided or generated)
├── data_loader/                      # Data loading utilities
├── evaluation/                       # Evaluation and metrics
├── model/                            # Model architectures and components
├── trainer/                          # Training logic
├── utils/                            # Utility functions
├── logger/                           # Logging and visualization utilities
├── base/                             # Base classes (models, trainers, loaders)
│
├── saved/                            # Saved models and logs
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── LICENSE                           # License file
│
└── .gitignore                        # Git ignore rules
```

---

## 📊 Dataset Format

- **Time series data**: CSV file  
  - Each column represents a variable  
  - Each row represents a time step  

- **Ground truth graph**: CSV file of tuples  

```
(i, j, t)
```

where:
- `i` = cause variable  
- `j` = effect variable  
- `t` = time lag  

---

## ▶️ Usage

Run experiments using:

```bash
python runner.py -c <config_file> -t <task_name>
```

### Example

```bash
python runner.py -c config/config_fMRI.json -t fMRI
```

---

### Available Tasks

- `demo`
- `v`
- `fork`
- `diamond`
- `mediator`
- `fMRI`
- `lorenz`

---

## ⚙️ Configuration

Configuration files are located in the `config/` directory.

Example:

```json
{
  "name": "Causal Discovery",
  "n_gpu": 1,
  "arch": {
    "type": "CausalInsight",
    "args": {
      "model_type": "lstm",
      "hidden_dim": 512,
      "lstm_hidden_dim": 512,
      "kernel_size": 5
    }
  },
  "data_loader": {
    "type": "TimeseriesDataLoader",
    "args":{
      "batch_size": 128,
      "time_step": 32,
      "output_window": 31,
      "feature_dim": 1,
      "output_dim": 1,
      "shuffle": true,
      "validation_split": 0.01,
      "num_workers": 2
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,
      "weight_decay": 0,
      "amsgrad": true
    }
  },
  "loss": "mse_loss",
  "metrics": [
    "mse_loss"
  ],
  "lr_scheduler": {
    "type": "StepLR",
    "args":{
      "step_size": 30,
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 10,
    "save_dir": "saved/",
    "save_freq": 1,
    "save_period": 1,
    "verbosity": 2,
    "monitor": "min val_loss",
    "early_stop": 10,
    "tensorboard": true,
    "mnt_metric": "val_loss",
    "mnt_mode": "min"
  }
}
```

## 🔄 Switching Between Models

You can switch between different backbone architectures by modifying the `model_type` field in your configuration file:

```json
"model_type": "cnn"
```

### Available options:

```json
"model_type": "linear"   // Linear baseline
"model_type": "mlp"      // Multilayer perceptron
"model_type": "cnn"      // Temporal convolutional model
"model_type": "lstm"     // Recurrent model (LSTM)
```

Each option changes only the prediction head, while the shared causal masking mechanism remains unchanged across all models.

After updating the config file, simply rerun the training or inference script to apply the selected architecture.

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@misc{redden2026causalinsight,
      title={Causal-INSIGHT: Probing Temporal Models to Extract Causal Structure}, 
      author={Benjamin Redden and Hui Wang and Shuyan Li},
      year={2026},
      eprint={2603.25473},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      note={Accepted at IJCNN 2026},
      url={https://arxiv.org/abs/2603.25473}, 
}
```

---

## 📜 License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.

---

## 🙏 Acknowledgements

This codebase builds on the PyTorch template by Victor Zhou:  
https://github.com/victoresque/pytorch-template