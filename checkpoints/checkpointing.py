import os
import torch
from datetime import datetime


# Funzione per il salvataggio del checkpoint
def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, checkpoint_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),  
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, checkpoint_path)
    print(f"Checkpoint salvato su: {checkpoint_path}")

# Load checkpoints
def load_checkpoint(model, optimizer, scheduler, run_name = False, checkpoint_dir="/content/drive/MyDrive/FL/FederatedLearningProject/checkpoints"):
    """Loads a model checkpoint if it exists.
       load: model state, optimizer state, schedulere statea

    Args:
        model: The PyTorch model.
        optimizer: The optimizer used for training.
        checkpoint_dir: The directory where to store/retrieve checkpoints.
        model_name: the name of the model

    Returns:
        start_epoch: The epoch number to resume training from.
        checkpoint_data: A dictionary containing the checkpoint informantions
    """

    if not run_name: 
        print(f"\nSpecify the model name you want to load \nnothing was done ")
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_checkpoint.pth")

    start_epoch = 1
    checkpoint_data = None
    if os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        start_epoch = checkpoint_data["epoch"] + 1
        print(f" Checkpoint caricato da {checkpoint_path}, riprendo da epoca {start_epoch}.")
    else:
        print(" Nessun checkpoint trovato, inizio da epoca 1.")

    return start_epoch, checkpoint_data