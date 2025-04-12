# FederatedLearningProject

This project explores federated model editing techniques.

## Project Structure

The repository is organized as follows:

```text
federated-model-editing/
│
├── 📁 data/                      # Dataset loaders and preprocessors
│   └── cifar100_loader.py
│
├── 📁 training/
│   ├── FL_training.py
│   └── centralized_training.py
│
├── 📁 experiments/
│   ├── baseline_centralized.ipynb   # Colab for centralized training
│   ├── baseline_federated.ipynb     # Colab for FL (FedAvg)
│   └── model_editing.ipynb          # Colab for sparse fine-tuning + model merging
│
├── 📁 checkpoints/
│   ├── checkpointing.py        # Save/load models
│   └── logger.py               # Logging experiments (e.g., CSV, WandB, etc.)
│
├── README.md                    # Project documentation (this file)
└── main.py                      # CLI wrapper to run training/evaluation
