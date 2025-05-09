import os
import torch


# Load checkpoints
def load_checkpoint(model, optimizer, checkpoint_path):
    """Loads a model checkpoint if it exists.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer used for training.
        checkpoint_dir: The directory where to store/retrieve checkpoints.
        model_name: the name of the model

    Returns:
        start_epoch: The epoch number to resume training from.
        checkpoint_data: A dictionary containing the checkpoint informantions
    """
    
    # PATH to store/retrieve checkpoints
    #checkpoint_dir = "/content/drive/MyDrive/FL/FederatedLearningProject/checkpoints"
    #os.makedirs(checkpoint_dir, exist_ok=True)
    #checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pth")

    start_epoch = 1
    checkpoint_data = None
    if os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        start_epoch = checkpoint_data["epoch"] + 1
        print(f" Checkpoint caricato da {checkpoint_path}, riprendo da epoca {start_epoch}.")
    else:
        print(" Nessun checkpoint trovato, inizio da epoca 1.")

    return start_epoch, checkpoint_data