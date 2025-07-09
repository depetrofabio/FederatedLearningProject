# FederatedLearningProject

This project explores federated model editing techniques.

## Project Structure

The repository is organized as follows:

```text
federated-model-editing/
│
├── 📁 data/                     
│   └── cifar100_loader.py          # CIFAR-100 dataset loading and preprocessing
│
├── 📁 training/                  
│   ├── FL_training.py                # Federated learning pipeline (FedAvg)
│   ├── centralized_training.py       # Centralized training baseline
│   └── FedMETA.py                    # Federated model editing with meta-learning
│
├── 📁 experiments/                 
│   ├── baseline_centralized.ipynb    # Centralized training experiment
│   ├── baseline_federated.ipynb      # Federated (FedAvg) experiment
│   ├── model_editing.ipynb           # Sparse fine-tuning and model merging (centralized)
│   ├── federated_model_editing.ipynb # Federated model editing experiments
│   └── FedMETA.ipynb                 # FedMETA evaluation and results      
│   └── models.py                     
├── 📁 checkpoints/                 
│   ├── checkpointing.py              # Model checkpoint saving/loading
│
└── 📄 README.md                      # Project overview and structure
```

## How to run experiments

The **Experiments** folder contains notebooks that implement the code used in our experiments. These notebooks make use of functions defined in the **training**, **data**, and **checkpoints** directories.

To run each experiment, execute the corresponding notebook, ensuring that the appropriate hyperparameters for each algorithm are correctly specified.


