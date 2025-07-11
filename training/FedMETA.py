import torch
import torch.nn as nn
import copy
from typing import List, Dict
from torch.utils.data import DataLoader
import numpy as np
import itertools
import wandb
from FederatedLearningProject.training.model_editing import compute_mask, SparseSGDM

###### TASK AGGREGATION FUNCTION ######
def aggregate_with_task_arithmetic(
    task_vectors: List[Dict[str, torch.Tensor]],
    initial_global_model: nn.Module
) -> nn.Module:
    """
    Aggregates client updates by summing their task vectors and applies the result
    to the initial global model.

    This method is designed for a federated learning scenario where each client
    trains a mutually exclusive subset of the model's parameters. Because the
    updates are non-overlapping, they can be directly summed without conflict.

    Args:
        task_vectors (List[Dict[str, torch.Tensor]]): A list where each element
            is a client's task vector. A task vector is a dictionary (like a
            state_dict) representing the change in weights, calculated as:
            (local_model_after_training - initial_global_model).
        initial_global_model (nn.Module): The original, shared global model that
            was sent to the clients at the beginning of the round.

    Returns:
        nn.Module: A new global model instance with the aggregated updates applied.
    """
    # Handle the edge case where no client updates are provided.
    if not task_vectors:
        print("Warning: No task vectors provided for aggregation. Returning the initial model.")
        return initial_global_model

    # --- Initialize the aggregated delta ---
    # Create a dictionary to store the sum of all task vectors. We initialize it
    # with tensors of zeros that have the same shape as the model's parameters.
    # Note(1): this will prevent errors if clients won't return all the keys (layer), 
    # but only layer where updates are present.
    aggregated_deltas = {                       # Note(2): aggregated_deltas will be stored in the same device as initial_global_model
        name: torch.zeros_like(param)
        for name, param in initial_global_model.named_parameters()
    }

    # --- Sum all the task vectors from the clients ---
    # Iterate through each client's contribution.
    for task_vector in task_vectors:
        # Iterate through each layer's change (delta) in that contribution.
        for name, delta_tensor in task_vector.items():
            if name in aggregated_deltas:
                # Add the client's delta to our running total.
                # Ensure tensors are on the same device before adding.
                aggregated_deltas[name] += delta_tensor.to(aggregated_deltas[name].device)

    # --- Apply the aggregated delta to a copy of the global model ---
    # We avoid to modify the object passed in input to this function as a good practice.
    # We expect that the function will be used this way: old_model = aggregate_with_task_arithmetic(__, old_model)
    new_global_model = copy.deepcopy(initial_global_model)

    # Use torch.no_grad() as we are manually changing the weights, not training.
    with torch.no_grad():
        # Iterate through the parameters of our new model instance.
        for name, param in new_global_model.named_parameters():
            if name in aggregated_deltas:
                # Apply the total update for this parameter.
                # param.add_() is in-place addition.
                param.add_(aggregated_deltas[name])

    return new_global_model



###### TRAIN CLIENT ######
def train_client(model, client_loader, optimizer_config, client_mask, criterion, device, batch_size=128, num_local_steps = 4): # num_local_epochs = J nel pdf e E nel paper
    local_model = copy.deepcopy(model).to(device)

    old_weights = {
        name: param.clone().detach().cpu()
        for name, param in local_model.state_dict().items()
    }

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
    
    with torch.no_grad(): # don't track these operations in the computation graph
        updated_weights = {
            name : param.cpu()
            for name, param in local_model.state_dict().items()
            if name in client_mask and client_mask[name].any # verifica che almeno un peso in quel tensore sia aggiornabile
        }

    delta_weights = {
        name: updated_weights[name] - old_weights[name]
        for name in updated_weights
    }

    return delta_weights, final_avg_client_loss, final_client_accuracy


##### TRAIN SERVER ######
def train_server(model, 
                 num_rounds, 
                 client_dataset, 
                 client_masks, # scegliere se passare l'optimizer
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
        delta_list = [] # lista di modelli 
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
            
            delta, client_loss, client_accuracy, = train_client(
                model=model,
                client_loader=client_loader,
                optimizer_config = optimizer_config,
                client_mask = client_masks[client_idx],
                num_local_steps=num_client_steps,
                device=device,
                criterion=criterion
            )

             # --- START: Added for debugging ---
            # Calculate the L2 norm of the client update (delta)
            l2_norm = 0.0
            with torch.no_grad():
                for name in delta:
                    l2_norm += torch.norm(delta[name].float(), p=2).item() ** 2
            l2_norm = l2_norm ** 0.5
            print(f"  Client {client_idx}: Update L2 Norm = {l2_norm:.4f}")
            # --- END: Added for debugging ---

            ### dizionario con key = layer modello, values = (indice dentro al tensore e il valore del parametro). Tupla con dentro due liste. 
            delta_list.append(delta)
            client_losses.append(client_loss)
            client_accuracies.append(client_accuracy)   

        ### UPDATE SERVER MODEL WITH NEW LOGIC
        model = aggregate_with_task_arithmetic(delta_list, model)

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
        'selected_clients': selected_clients_history
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



##### MASK AGGREGATION #####
# output: global mask 
def aggregate_masks(local_masks, threshold_ratio=0.8):
    """
    Aggrega n maschere locali in una maschera globale.

    Args:
        local_masks (list of dict): Lista di maschere locali (bool tensor per parametro).
        threshold_ratio (float): Soglia minima (es. 0.5 significa 50% dei client).

    Returns:
        dict: Maschera globale con stessi nomi dei parametri.
    """
    if not local_masks:
        raise ValueError("La lista delle maschere locali è vuota!")

    num_clients = len(local_masks)
    threshold = int(num_clients * threshold_ratio)

    # Inizializza l'accumulatore a zeri interi
    agg_mask = {
        name: torch.zeros_like(mask, dtype=torch.int32)
        for name, mask in local_masks[0].items()
    }

    # Somma le maschere locali
    for cm in local_masks:
        for name in agg_mask:
            agg_mask[name] += cm[name].int()

    # Crea la maschera finale con soglia
    final_mask = {
        name: (agg_mask[name] >= threshold)
        for name in agg_mask
    }

    return final_mask

# NEW VERSION
def aggregate_masks_by_sparsity(
    local_masks: List[Dict[str, torch.Tensor]],
    sparsity_target: float
) -> Dict[str, torch.Tensor]:
    """
    Aggrega le maschere locali per creare una maschera globale che raggiunge
    un obiettivo di sparsità specifico, calcolando la soglia tramite quantile.

    Args:
        local_masks (List[Dict[str, torch.Tensor]]): Lista di maschere locali.
            Si assume che ogni maschera nel dizionario abbia le stesse chiavi.
        sparsity_target (float): La frazione desiderata di parametri da mascherare
            (es. 0.9 per il 90% di zeri).

    Returns:
        Dict[str, torch.Tensor]: La maschera globale finale con la sparsità desiderata.
    """
    if not local_masks:
        raise ValueError("La lista delle maschere locali è vuota!")
    if not (0.0 <= sparsity_target < 1.0):
        raise ValueError("Sparsity target deve essere compreso tra [0.0, 1.0)")

    # --- Somma i "voti" da tutte le maschere locali (logica originale) ---
    # Inizializza l'accumulatore a zeri interi, usando le chiavi della prima maschera
    # come riferimento (assumendo che tutte le maschere abbiano le stesse chiavi).
    agg_mask_votes = {
        name: torch.zeros_like(mask, dtype=torch.int32)
        for name, mask in local_masks[0].items()
    }

    # Somma i valori booleani (True=1, False=0) di ogni maschera locale.
    for client_mask in local_masks:
        for name in agg_mask_votes:
            agg_mask_votes[name] += client_mask[name].int()

    # --- Trova la soglia dinamica tramite quantile ---
    # Raccogli tutti i valori di voto da tutti i layer in un unico tensore 1D.
    all_votes = torch.cat([votes.flatten() for votes in agg_mask_votes.values()])
    
    total_params = all_votes.numel()
    
    # Calcola quanti parametri devono rimanere attivi (non mascherati).
    num_to_keep = max(1, int(total_params * (1.0 - sparsity_target)))

    if num_to_keep >= total_params:
        # Se la sparsità è 0, teniamo tutti i parametri con almeno un voto.
        threshold = 1
    else:
        # Ordina i voti in ordine decrescente e trova il valore di soglia.
        # Il valore del k-esimo elemento più grande diventa la nostra soglia.
        # num_to_keep - 1 perché gli indici partono da 0.
        threshold = torch.sort(all_votes, descending=True).values[num_to_keep - 1]
    
    print(f"Sparsity Target: {sparsity_target:.2f} -> Keeping {num_to_keep}/{total_params} params.")
    print(f"Calculated Threshold (min votes to be kept): {threshold.item()}")

    # --- Crea la maschera finale usando la soglia calcolata ---
    # Un parametro viene tenuto (maschera = 1) se il suo numero di voti è >= della soglia.
    final_mask = {
        name: (votes >= threshold)      # broadcasting
        for name, votes in agg_mask_votes.items()
    }
    
    # Verifica la sparsità effettiva ottenuta (dovrebbe essere molto vicina al target)
    final_kept = sum((m.sum().item() for m in final_mask.values()))
    final_sparsity = 1.0 - (final_kept / total_params) if total_params > 0 else 0
    print(f"Achieved Sparsity: {final_sparsity:.4f}")

    return final_mask


##### PARAMETER PARTITION FUNCTION #####
def distribution_function(final_mask, number_clients):
    '''
    final_mask: dict of tensors with 1s and 0s (global mask)
    unmasked_params: total number of 1s in final_mask
    number_clients: number of clients to partition the unmasked parameters

    Returns:
        client_masks: list of length = number_clients
                      each element is a dict (same keys as final_mask)
                      with 1s in unique positions (disjoint among clients)
    '''

    _, _, unmasked_params =  count_masked_params(final_mask)
    #print(f"Totale parametri in final_mask: {total_params}")

    base_params_per_client = unmasked_params // number_clients
    remainder = unmasked_params % number_clients

    client_masks = [dict() for _ in range(number_clients)] # one dict (local mutual exclusive mask per client

    # Costruiamo la lista di tutte le posizioni degli 1 nella maschera globale
    all_1_positions = []
    for key, mask_tensor in final_mask.items():
        ones_indices = torch.nonzero(mask_tensor.flatten(), as_tuple=False).squeeze() # ones_indices dim = [num_ones, 1]
        # ones_indices può essere un tensor 1D o 0D se solo 1 elemento
        if ones_indices.ndim == 0:
            ones_indices = ones_indices.unsqueeze(0)
        for idx in ones_indices.tolist():
            all_1_positions.append((key, idx))  # avremo key = layer, value = indice nel tensore dell'1

    # Shuffle
    torch.manual_seed(0) # for reproducibility
    perm = torch.randperm(len(all_1_positions)).tolist()  # crea permutazione casuale degli interi da 0 a numero_tot_1
    all_1_positions = [all_1_positions[i] for i in perm]  # shuffle

    start_idx = 0
    for client_id in range(number_clients):
        count = base_params_per_client + (1 if client_id == number_clients - 1 else 0) # if the client is the last client assign one param more
        subset = all_1_positions[start_idx:start_idx+count] # assign (key = layer, value = idx) pairs to clients randomly
        start_idx += count

        for key, flat_idx in subset:
            shape = final_mask[key].shape
            if key not in client_masks[client_id]:  # if a key (layer) is not present in a client add it with all zeros
                client_masks[client_id][key] = torch.zeros_like(final_mask[key], dtype=torch.bool)
            idx_unravel = torch.unravel_index(torch.tensor(flat_idx), shape) # nota che la maschera di un paramtro (che fin'ora ho chiamato layer) può essere un tensore anche 2D, con unravel ricostruiamo le coordinate 2D
            client_masks[client_id][key][idx_unravel] = True

    # Per sicurezza, per ogni client assicuriamo che tutte le chiavi siano presenti:
    for client_mask in client_masks:
        for key in final_mask.keys():
            if key not in client_mask:
                client_mask[key] = torch.zeros_like(final_mask[key], dtype=torch.bool)

    return client_masks # lista di maschere dei client (lista di dizionari)

## util for distribution_function
# conta i parameteri totali - masked - non masked
def count_masked_params(mask):
    total_params = 0
    masked_params = 0

    for key, mask_tensor in mask.items():
        total_params += mask_tensor.numel()          # total number of elements in this parameter
        masked_params += (mask_tensor == 0).sum().item()  # count how many elements are zero (masked)

    unmasked_params = total_params - masked_params

    print(f"Total parameters: {total_params}")
    print(f"Masked parameters (zeros): {masked_params}")
    print(f"Unmasked parameters (ones): {unmasked_params}")

    return total_params, masked_params, unmasked_params


def log_to_wandb_fedavg(round, client_avg_loss, client_avg_accuracy, server_val_loss, server_val_accuracy):
    wandb.log({
        "client_avg_loss": client_avg_loss,
        "client_avg_accuracy": client_avg_accuracy,
        "server_val_loss": server_val_loss,
        "server_val_accuracy": server_val_accuracy,
        "round":round
    }, step=round)