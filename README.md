# FederatedLearningProject

This project explores federated model editing techniques.

## Project Structure

The repository is organized as follows:

```text
federated-model-editing/
│
├── 📁 data/                         # Dataset loaders and preprocessing utilities
│   └── cifar100_loader.py          # CIFAR-100 loading and preprocessing
│
├── 📁 training/                     # Training pipelines
│   ├── FL_training.py              # Standard FedAvg implementation
│   ├── centralized_training.py     # Centralized training baseline
│   └── FedMETA.py                  # Federated model editing via meta-learning
│
├── 📁 experiments/                 # Colab notebooks for reproducibility
│   ├── baseline_centralized.ipynb  # Centralized training experiment
│   ├── baseline_federated.ipynb    # Federated (FedAvg) experiment
│   ├── model_editing.ipynb         # Sparse fine-tuning + model merging
│   └── FedMETA.ipynb               # FedMETA evaluation and experiments
│
├── 📁 checkpoints/                 # Model saving and logging
│   ├── best_model.pth              # Final or best-performing model checkpoint
│   ├── checkpointing.py            # Save/load model state dictionaries
│   └── logger.py                   # Logging utility (CSV, WandB, etc.)
│
└── 📄 README.md                    # This file


