from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import copy
# from training.centralized_training import train_epoch, validate_epoch
from FederatedLearningProject.training.centralized_training import train_epoch, validate_epoch
from torch.optim.lr_scheduler import CosineAnnealingLR

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


# come posizionare lo scheduler nel FL setting?
def train_client(model, client_loader, global_scheduler, global_optimizer, criterion, device, num_local_epochs, debug): # num_local_epochs = J nel pdf e E nel paper
    local_model = copy.deepcopy(model).to(device)
    if debug:
        debug_model(local_model)
    
    
    # print(f"Il device su cui è la copia del local_model è {local_model.get_device()}")
    # local_model.train()             # il modello dovrebbe già essere nelle condizioni adatte, non va settato tutto su train 
                                      # per evitare drop out dei layer bloccati
                                    
    # questo crea un optimizer per i client con gli hyperparameters del global optimizer (quello del server),
    #  


    #local_optimizer = type(optimizer)(local_model.parameters(), **optimizer.defaults)
    #local_scheduler = type(scheduler)(local_optimizer, **scheduler.state_dict())

    local_optimizer = type(global_optimizer)(local_model.parameters(), **global_optimizer.defaults)

    local_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        local_optimizer,
        T_max=num_local_epochs, # T_max = numero di epoche locali
    )


    client_accuracies_epoch = []
    client_losses_epoch = []

    for _ in range(num_local_epochs):
        client_accuracy, avg_client_loss = train_epoch(
            model=local_model, 
            train_loader=client_loader,
            scheduler=local_scheduler,
            optimizer=local_optimizer,
            criterion=criterion, 
            device=device
        )
        client_accuracies_epoch.append(client_accuracy)
        client_losses_epoch.append(avg_client_loss)
        local_scheduler.step() # ricordare di mettere lo scheduler.step() ogni EPOCA anche nel centralized

    final_avg_client_loss = client_losses_epoch[-1] if client_losses_epoch else 0
    final_client_accuracy = client_accuracies_epoch[-1] if client_accuracies_epoch else 0

    return local_model, final_avg_client_loss, final_client_accuracy


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


def train_server(model, num_rounds, client_dataset, optimizer, scheduler, device, val_loader, n_epochs_log=5 ,num_clients=100, num_client_epochs=4, frac=0.1, criterion=nn.CrossEntropyLoss, batch_size=128, debug = False):
    train_losses = []
    val_accuracies = []
    selected_clients_history = [] # lista dei clients selezionati (print ad ogni round di comunicazione)

    for round in range(num_rounds):
        client_models = [] # lista di modelli 
        client_losses = [] 
        client_accuracies = []
        client_sizes = [] 

        # seleziona i clients come nell'algoritmo FedAvg (paper di McMahan)
        m = max(int(num_clients * frac), 1)
        idx_clients = np.random.choice(range(num_clients), m, replace=False) 
        selected_clients_history.append(idx_clients)

        for client_idx in idx_clients:
            client_loader = DataLoader(client_dataset[client_idx], batch_size=batch_size, shuffle=True)
            client_size = len(client_dataset[client_idx])
            client_sizes.append(client_size)

            client_model, client_loss, client_accuracy = train_client(
                model=model,
                client_loader=client_loader,
                num_local_epochs=num_client_epochs,
                device=device,
                global_scheduler=scheduler,
                global_optimizer=optimizer,
                criterion=criterion,
                debug = debug,
            )

            client_models.append(client_model.state_dict())
            client_losses.append(client_loss)
            client_accuracies.append(client_accuracy)

        updated_weights = average_weights(client_models, client_sizes)
        model.load_state_dict(updated_weights)

        avg_loss = sum(client_losses) / len(client_losses)
        train_losses.append(avg_loss)

        if (round + 1) % n_epochs_log == 0:
            val_loss, val_acc = val(model, val_loader, device, criterion) 
            val_accuracies.append(val_acc)
            
            print(f"\nRound {round+1}/{num_rounds}")
            print(f"Selected Clients: {idx_clients}")
            print(f"Avg Client Loss: {avg_loss:.4f} | Avg Client Accuracy: {sum(client_accuracies)/len(client_accuracies):.2f}%")
            print(f"Evaluation Loss: {val_loss:.4f} | Test Accuracy: {val_acc:.2f}%")
            print("-" * 50)

    return {
        'model': model,
        'train_losses': train_losses,
        'test_accuracies': val_accuracies,
        'selected_clients': selected_clients_history
    }

def val(model, val_loader, device, criterion):
    # DOMANDA : bisogna testare il la copia del modello sui singoli client?

    # copia del modello nell'evaluation, altrimenti facendo model.eval() si metterebbe tutto il modello in .eval() 
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

