import os
import copy
import itertools
import numpy as np

# Typing
from typing import Any, Dict, List

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optimizer & Scheduler
from torch.optim import SGD as SGDM
from torch.optim.lr_scheduler import CosineAnnealingLR

# Custom imports
from FederatedLearningProject.checkpoints.checkpointing import save_checkpoint_fedavg
from FederatedLearningProject.training.model_editing import compute_mask, SparseSGDM  # removed '.py' from import

# Weights & Biases
import wandb

def debug_model(model: nn.Module, model_name: str = "Model"):
    """
    Prints debugging information about a PyTorch model.

    Information includes:
    - Overall device of the first parameter (indicative of model's primary device).
    - For each named parameter:
        - Full parameter name.
        - Device of the parameter.
        - Whether the parameter requires gradients (is frozen or not).
        - Inferred block index if the name matches a ViT-like structure.
    """
    print(f"\n--- Debugging {model_name} ---")

    # Check overall model device (based on the first parameter)
    try:
        first_param_device = next(model.parameters()).device
        print(f"{model_name} is primarily on device: {first_param_device}")
    except StopIteration:
        print(f"{model_name} has no parameters.")
        return

    print("\nParameter Details (Name | Device | Requires Grad? | Inferred Block):")
    for name, param in model.named_parameters():
        device = param.device
        requires_grad = param.requires_grad
        
        block_info = "N/A"
        # Try to infer block index for ViT-like models
        if "blocks." in name:
            try:
                # e.g., name = "blocks.0.attn.qkv.weight"
                block_idx_str = name.split("blocks.")[1].split(".")[0]
                if block_idx_str.isdigit():
                    block_info = f"Block {block_idx_str}"
            except IndexError:
                block_info = "Block (parse error)"

        print(f"- {name:<50} | {str(device):<10} | {str(requires_grad):<15} | {block_info}")
    
    # You can add more specific checks here, e.g., for model mode (train/eval)
    print(f"{model_name} is in {'training' if model.training else 'evaluation'} mode.")
    print(f"--- End Debugging {model_name} ---\n")

def average_weights(weights: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
    total_size = sum(client_sizes) # calcola il numero totale di samples tra tutti i clients che partecipano al round
    weights_avg = copy.deepcopy(weights[0]) 

    for key in weights_avg.keys(): # itera tra tutte lei "key" dove per "Key" si intende l'insieme di parametri di un layer
        # primo step dell'algoritmo: inizializzazione dei pesi attraverso i parametri del primo client
        weights_avg[key] = weights_avg[key] * (client_sizes[0] / total_size) # assicura che i parametri del primo client siano scalati
        
    for i in range(1, len(weights)): # iterazione sugli altri client 
        for key in weights_avg.keys():
            weights_avg[key] += weights[i][key] * (client_sizes[i] / total_size)
            
    return weights_avg
 

def log_to_wandb_fedavg(round, client_avg_loss, client_avg_accuracy, server_val_loss, server_val_accuracy):
    wandb.log({
        "client_avg_loss": client_avg_loss,
        "client_avg_accuracy": client_avg_accuracy,
        "server_val_loss": server_val_loss,
        "server_val_accuracy": server_val_accuracy,
        "round":round
    }, step=round)


###### TRAIN CLIENT ######
def train_client_model_editing(model, client_loader, optimizer_config, client_mask, criterion, device, batch_size=128, num_local_steps = 4): # num_local_epochs = J nel pdf e E nel paper
    local_model = copy.deepcopy(model).to(device)

    data_iterator = itertools.cycle(client_loader)
    local_optimizer = SparseSGDM(
        local_model.named_parameters(),
        masks=client_mask,
        lr=optimizer_config.get("lr"),
        weight_decay=optimizer_config.get("weight_decay"),
        momentum=optimizer_config.get("momentum")
    )

    client_accuracies_epoch = []
    client_losses_epoch = []
    
    # Initialize accumulators for this client's local training
    accumulated_loss = 0.0
    accumulated_correct = 0
    accumulated_total_samples = 0

    for step in range(num_local_steps):
        images, labels = next(data_iterator)
        images, labels = images.to(device), labels.to(device)

        local_optimizer.zero_grad()
        outputs = local_model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        local_optimizer.step()

        accumulated_loss += loss.item() * images.size(0)
        _,predicted = torch.max(outputs, 1)
        accumulated_total_samples += labels.size(0)
        accumulated_correct += (predicted == labels).sum().item()
 
    if accumulated_total_samples > 0:
        client_accuracy = 100 * accumulated_correct / accumulated_total_samples
        avg_client_loss = accumulated_loss / accumulated_total_samples
    else:
        client_accuracy = 0.0
        avg_client_loss = 0.0

    client_accuracies_epoch.append(client_accuracy)
    client_losses_epoch.append(avg_client_loss)

    final_avg_client_loss = client_losses_epoch[-1] if client_losses_epoch else 0
    final_client_accuracy = client_accuracies_epoch[-1] if client_accuracies_epoch else 0
    


    return local_model, final_avg_client_loss, final_client_accuracy


##### TRAIN SERVER ######
def train_server_model_editing(model, 
                 num_rounds, 
                 client_dataset, 
                 client_masks, 
                 optimizer_config,
                 device, 
                 val_loader, 
                 checkpoint_path,
                 n_rounds_log=5,
                 num_clients=100, 
                 num_client_steps=4,
                 frac=0.1,
                 criterion=nn.CrossEntropyLoss(),
                 batch_size=64,
                 debug = False, 
                 model_name="dino_vits16"):
    train_losses = []
    val_accuracies = []
    selected_clients_history = []
    generator = torch.Generator()
    generator.manual_seed(42)


    for round in range(num_rounds):
        client_models = [] # lista di modelli 
        client_losses = [] 
        client_accuracies = []
        client_sizes = [] 

        m = max(int(num_clients * frac), 1)
        idx_clients = np.random.choice(range(num_clients), m, replace=False) 
        selected_clients_history.append(idx_clients)

        print(f"\n--- Round {round+1}/{num_rounds} ---") # Added for clarity

        for client_idx in idx_clients:
            client_loader = DataLoader(client_dataset[client_idx], batch_size=batch_size, shuffle=True, generator=generator) 
            client_size = len(client_dataset[client_idx])
            client_sizes.append(client_size)
            
            client_model, client_loss, client_accuracy, = train_client_model_editing(
                model=model,
                client_loader=client_loader,
                optimizer_config = optimizer_config,
                client_mask = client_masks[client_idx],
                num_local_steps=num_client_steps,
                device=device,
                criterion=criterion
            )

            client_models.append(client_model.state_dict())
            client_losses.append(client_loss)
            client_accuracies.append(client_accuracy)   

        ### UPDATE SERVER MODEL WITH NEW LOGIC
        updated_weights = average_weights(client_models, client_sizes)
        model.load_state_dict(updated_weights)

        avg_loss = sum(client_losses) / len(client_losses)
        train_losses.append(avg_loss)
        
        if (round + 1) % n_rounds_log == 0:

            client_avg_accuracy_for_log = sum(client_accuracies) / len(client_accuracies)

            val_loss, val_acc = val(model, val_loader, device, criterion) 
            val_accuracies.append(val_acc)
            
            log_to_wandb_fedavg(round=round, client_avg_loss = avg_loss, client_avg_accuracy= client_avg_accuracy_for_log, server_val_accuracy=val_acc, server_val_loss=val_loss)

            # --- Save checkpoint ---
            # checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpointFINAL.pth")

            print(f"\nRound {round+1}/{num_rounds}")
            print(f"Selected Clients: {idx_clients}")
            print(f"Avg Client Loss: {avg_loss:.4f} | Avg Client Accuracy: {sum(client_accuracies)/len(client_accuracies):.2f}%")
            print(f"Evaluation Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
            print("-" * 50)

    return {
        'model': model,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'selected_clients': selected_clients_history,
        'client_accuracies': client_accuracies
    }

def train_client(model, client_loader, optimizer_config, criterion, device, batch_size=128, num_local_steps = 4): # num_local_epochs = J nel pdf e E nel paper
    local_model = copy.deepcopy(model).to(device)

    data_iterator = itertools.cycle(client_loader)
    local_optimizer = SGD(
        local_model.named_parameters(),
        lr=optimizer_config.get("lr"),
        weight_decay=optimizer_config.get("weight_decay"),
        momentum=optimizer_config.get("momentum")
    )

    client_accuracies_epoch = []
    client_losses_epoch = []
    
    # Initialize accumulators for this client's local training
    accumulated_loss = 0.0
    accumulated_correct = 0
    accumulated_total_samples = 0

    for step in range(num_local_steps):
        images, labels = next(data_iterator)
        images, labels = images.to(device), labels.to(device)

        local_optimizer.zero_grad()
        outputs = local_model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        local_optimizer.step()

        accumulated_loss += loss.item() * images.size(0)
        _,predicted = torch.max(outputs, 1)
        accumulated_total_samples += labels.size(0)
        accumulated_correct += (predicted == labels).sum().item()
 
    if accumulated_total_samples > 0:
        client_accuracy = 100 * accumulated_correct / accumulated_total_samples
        avg_client_loss = accumulated_loss / accumulated_total_samples
    else:
        client_accuracy = 0.0
        avg_client_loss = 0.0

    client_accuracies_epoch.append(client_accuracy)
    client_losses_epoch.append(avg_client_loss)

    final_avg_client_loss = client_losses_epoch[-1] if client_losses_epoch else 0
    final_client_accuracy = client_accuracies_epoch[-1] if client_accuracies_epoch else 0
    


    return local_model, final_avg_client_loss, final_client_accuracy

def train_server(model, 
                 num_rounds, 
                 client_dataset, 
                 optimizer_config,
                 device, 
                 val_loader, 
                 checkpoint_path,
                 n_rounds_log=5,
                 num_clients=100, 
                 num_client_steps=4,
                 frac=0.1,
                 criterion=nn.CrossEntropyLoss(),
                 batch_size=64,
                 debug = False, # Added checkpoint_dir="/content/drive/MyDrive/FL/FederatedLearningProject/checkpoints"
                 model_name="dino_vits16"): # Added
    train_losses = []
    val_accuracies = []
    selected_clients_history = []
    generator = torch.Generator()
    generator.manual_seed(42)


    for round in range(num_rounds):
        client_models = [] # lista di modelli 
        client_losses = [] 
        client_accuracies = []
        client_sizes = [] 

        m = max(int(num_clients * frac), 1)
        idx_clients = np.random.choice(range(num_clients), m, replace=False) 
        selected_clients_history.append(idx_clients)

        print(f"\n--- Round {round+1}/{num_rounds} ---") # Added for clarity

        for client_idx in idx_clients:
            client_loader = DataLoader(client_dataset[client_idx], batch_size=batch_size, shuffle=True, generator=generator) 
            client_size = len(client_dataset[client_idx])
            client_sizes.append(client_size)
            
            client_model, client_loss, client_accuracy, = train_client(
                model=model,
                client_loader=client_loader,
                optimizer_config = optimizer_config,
                num_local_steps=num_client_steps,
                device=device,
                criterion=criterion
            )

            client_losses.append(client_loss)
            client_accuracies.append(client_accuracy)   

        ### UPDATE SERVER MODEL WITH NEW LOGIC
        updated_weights = average_weights(client_models, client_sizes)
        model.load_state_dict(updated_weights)

        avg_loss = sum(client_losses) / len(client_losses)
        train_losses.append(avg_loss)
        
        if (round + 1) % n_rounds_log == 0:

            client_avg_accuracy_for_log = sum(client_accuracies) / len(client_accuracies)

            val_loss, val_acc = val(model, val_loader, device, criterion) 
            val_accuracies.append(val_acc)
            
            log_to_wandb_fedavg(round=round, client_avg_loss = avg_loss, client_avg_accuracy= client_avg_accuracy_for_log, server_val_accuracy=val_acc, server_val_loss=val_loss)

            # --- Save checkpoint ---
            # checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpointFINAL.pth")

            print(f"\nRound {round+1}/{num_rounds}")
            print(f"Selected Clients: {idx_clients}")
            print(f"Avg Client Loss: {avg_loss:.4f} | Avg Client Accuracy: {sum(client_accuracies)/len(client_accuracies):.2f}%")
            print(f"Evaluation Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
            print("-" * 50)

    return {
        'model': model,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'selected_clients': selected_clients_history,
        'client_accuracies': client_accuracies
    }

# util for train server
def val(model, val_loader, device, criterion):
    local_model = copy.deepcopy(model) 
    local_model.eval()  
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = local_model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy