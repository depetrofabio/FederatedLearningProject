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
├── 📁 models/                    # Model definitions and utilities
│   ├── vit.py                  # Vision Transformer (DINO ViT-S/16)
│   └── utils.py
│
├── 📁 training/
│   ├── centralized.py          # Centralized training loop
│   ├── federated.py            # Federated learning logic (FedAvg)
│   └── sparse_training.py      # SparseSGDM-based training logic
│
├── 📁 editing/
│   ├── mask_calibration.py     # Gradient mask calibration (Fisher info, etc.)
│   └── sparse_sgdm.py          # SparseSGDM optimizer
│
├── 📁 experiments/
│   ├── baseline_centralized.ipynb   # Colab for centralized training
│   ├── baseline_federated.ipynb     # Colab for FL (FedAvg)
│   └── model_editing.ipynb          # Colab for sparse fine-tuning + model merging
│
├── 📁 utils/
│   ├── checkpointing.py        # Save/load models
│   └── logger.py               # Logging experiments (e.g., CSV, WandB, etc.)
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation (this file)
└── main.py                      # CLI wrapper to run training/evaluation
