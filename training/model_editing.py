import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import copy
import time


######## SPARSE SGDM ##########
# implementation of a personalized optimizer

from torch.optim.optimizer import Optimizer, required
from typing import Dict, Iterable, Optional

class SparseSGDM(Optimizer):
    r"""Implements Stochastic Gradient Descent with Momentum and a sparsity mask.

    This optimizer is an extension of the standard `torch.optim.SGD`. It is designed
    for model editing and sparse fine-tuning tasks where only a specific subset of
    model weights should be updated.

    The core functionality is the introduction of a `mask` for each parameter.
    This mask is a binary tensor (containing 0s and 1s) with the same shape as the
    parameter it corresponds to. During the update step, the calculated gradient
    and momentum (the final update vector) is element-wise multiplied by this mask.
    This operation effectively zeroes out the updates for weights where the mask
    has a value of 0, thus "freezing" them.

    This class is implemented by re-creating the `step` method of the standard
    SGDM optimizer to correctly inject the masking operation *after* the momentum
    calculation, which is crucial for correctly freezing weights that have a
    non-zero momentum history. (Note: for us is useless)

    Args:
        params (iterable): An iterable of (name, parameter) tuples, typically from
            `model.named_parameters()`.
        masks (dict): A dictionary mapping parameter names (str) to mask tensors.
            The optimizer will only update parameters that have a corresponding
            entry in this dictionary.
        lr (float): The learning rate. It's a required argument.
        momentum (float, optional): The momentum factor (default: 0).
        weight_decay (float, optional): The weight decay (L2 penalty) factor (default: 0).
        dampening (float, optional): A dampening factor for momentum (default: 0).
        nesterov (bool, optional): Enables Nesterov Accelerated Gradient (NAG) (default: False).
    """

    # The init method is used to initialize the optimizer's internal state. 
    # In this method, we define the hyperparameters of the optimizer and set the internal state
    def __init__(self, params: Iterable[tuple[str, torch.Tensor]], 
                 masks: Dict[str, torch.Tensor], 
                 lr: float = required, momentum: float = 0, dampening: float = 0,
                 weight_decay: float = 0, nesterov: bool = False):
        
        # --- Argument Validation ---
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening.")

        # --- Aligning Parameters with Masks using Names ---
        # We iterate through the named parameters and use the 
        # name to look up the corresponding mask from the dictionary. 
        # This avoids any issues with parameter/mask ordering.

        # What happens in practice:
        # -> We create a separate param_group for each individual parameter (tensor) that we want to optimize.
        # -> Inside each group, we store not only the parameter itself ('params': [param]) but also our custom information: 
        #    the corresponding sparsity mask ('mask': mask).
        # This allows the optimizer's step function to loop through each parameter, easily access its specific mask, 
        # and apply the correct sparse update logic.
        param_groups = []
        for name, param in params:   # model.named_params
            # We only create a parameter group for parameters that are included in the mask dictionary.
            # This allows you to naturally control which layers are optimized.
            if name in masks:
                mask = masks[name]
                if param.shape != mask.shape:
                    raise ValueError(
                        f"Shape of parameter '{name}' ({param.shape}) does not match "
                        f"shape of its mask ({mask.shape})."
                    )
                # Each group contains the parameter itself and its mask.
                # I added the name for debugging if needed
                param_groups.append({'params': [param], 'mask': mask, 'name': name})

        if not param_groups:
            raise ValueError("No parameters to optimize. Make sure the names in `params` match the keys in the `masks` dictionary.")

        # --- Default Hyperparameters ---
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        
        # --- Initialize the Parent Optimizer ---
        # we pass param groups (list of dict) which encode all information about the mask,
        # this way we'll be able to use it in the step()
        # Il costruttore della classe genitore fa una cosa molto intelligente per noi: 
        #       > Prende la lista param_groups e la salva internamente come self.param_groups. 
        #       > Poi, scorre ogni singolo "pacchetto" (dizionario) nella lista.
        #       > Per ogni pacchetto, controlla quali iperparametri dal dizionario defaults mancano.
        #       > Copia tutti gli iperparametri mancanti da defaults direttamente dentro al pacchetto.
        super().__init__(param_groups, defaults)  # calling the constructor (__init__ insomma) of torch.optim.Optimizer 

# ---  Use the parent method to resume a checkpoint ---
    def __setstate__(self, state): 
        super().__setstate__(state)
        for group in self.param_groups:         # ensures backward compatibility when loading state dicts that might omit the nesterov flag.
            group.setdefault('nesterov', False)

# --- Mask logic implementation ---
    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.

        This is a lightweight version of the step function that omits the 'closure'
        logic, as it is not required for optimizers like SGD and its variants.
        """
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []   # delta-p will store the gradient tensor for each parameter in params_with_grad
            momentum_buffer_list = [] # store momentum
            
            # --- Retrieve Group-Specific Settings ---
            # For the current parameter group being processed, we unpack all its
            # specific settings. These values are present in the `group` dictionary
            # because the parent 'Optimizer' class automatically populated them
            # from the 'defaults' dictionary during initialization.
            # (Vedi sopra quando chiamo super().__init()__(param_groups, defaults))
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            lr = group['lr']
            # This is our custom setting, which we added to the group ourselves.
            mask = group['mask']

            # this is the standard way of writing this part, but actually we have a 
            # single tensor in group['params'], so it is equivalent to: p = group['params'][0]
            # => the loop runs only once
            for p in group['params']:     # equivalent to p = group['params][0]
                if p.grad is not None:
                    params_with_grad.append(p)  # append the tensor p
                    d_p_list.append(p.grad)     # append gradient of the tensor. p.grad contains the grad. of the loss f. w.r.t the most recent backward pass (result of loss.backward())
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer']) # If a momentum buffer does exist from a previous step, we retrieve it from the state dictionary and add it to our momentum_buffer_list

            for i, param in enumerate(params_with_grad):  # again in our case this runs only once
                d_p = d_p_list[i]                         # equivalent to d_p = d_p_list[0]
                
                # -- See SGDM pseudocode --
                if weight_decay != 0:
                    # vectorized operation: g_t <- g_t + lambda*theta_(t-1)
                    d_p = d_p.add(param, alpha=weight_decay) # if enabled, weight_decay will add fraction of the weight to the loss to prevent it growing to large
                
                if momentum != 0:
                    buf = momentum_buffer_list[i]
                    if buf is None:  # at the first step, we must initialize momentum
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)  # b_t <- mu*b_(t-1) + (1-tau)*g_t
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                d_p_masked = d_p * mask.to(d_p.device)  # elem-wise mul
                param.add_(d_p_masked, alpha=-lr)       # theta_t <- theta_(t-1) - gamma*masked_update

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                self.state[p]['momentum_buffer'] = momentum_buffer
                
                



#-------------------------------------------------------------------------------
# PLOT: masked layers

def plot_all_layers_mask_sparsity(final_mask):
    layer_names = []
    active_percentages = []

    for name, mask in final_mask.items():
        total_params = mask.numel()
        active_params = (mask == 1).sum().item()
        percentage = 100 * active_params / total_params

        if percentage > 0: # Only plot the layers that have some ones
          layer_names.append(name)
          active_percentages.append(percentage)

    # Plot
    plt.figure(figsize=(6, max(6, len(layer_names) * 0.2)))
    plt.barh(layer_names, active_percentages)
    plt.xlabel("Percentuale di pesi attivi (mask = 1)")
    plt.title("Sparsità residua per layer dopo pruning")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()


def plot_qkv_weight_bias_sparsity(final_mask):
    labels = []
    percentages = []
    colors = []

    for name, mask in final_mask.items():
        if "qkv.weight" in name or "qkv.bias" in name:
            # Separazione Q, K, V
            size = mask.shape[0] // 3
            q_mask = mask[:size]
            k_mask = mask[size:2*size]
            v_mask = mask[2*size:]

            if mask.dim() == 2:  # weight: shape [3*D, D]
                q_label = name.replace(".qkv.weight", "") + "_Q_weight"
                k_label = name.replace(".qkv.weight", "") + "_K_weight"
                v_label = name.replace(".qkv.weight", "") + "_V_weight"
            else:  # bias: shape [3*D]
                q_label = name.replace(".qkv.bias", "") + "_Q_bias"
                k_label = name.replace(".qkv.bias", "") + "_K_bias"
                v_label = name.replace(".qkv.bias", "") + "_V_bias"

            for lbl, m in zip([q_label, k_label, v_label], [q_mask, k_mask, v_mask]):
                active = (m == 1).sum().item()
                total = m.numel()
                perc = 100 * active / total
                labels.append(lbl)
                percentages.append(perc)
                colors.append("steelblue" if "weight" in lbl else "orange")

    # Plot
    plt.figure(figsize=(14, max(6, len(labels)*0.3)))
    bars = plt.barh(labels, percentages, color=colors)
    plt.xlabel("Percentuale di pesi attivi (mask = 1)")
    plt.title("Sparsità componenti Q/K/V - weight e bias")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def plot_changed_weights_percentage(before_weights, after_model, eps=1e-6):
    """
    Plotta la percentuale di pesi modificati per ogni layer.
    
    Args:
        before_weights (dict): dizionario {name: param.data.clone()} salvato prima del training.
        after_model (nn.Module): il modello dopo il training.
        eps (float): soglia minima di differenza per considerare un peso "cambiato".
    """
    changed_percentages = []
    layer_names = []

    for name, param in after_model.named_parameters():
        if name in before_weights:
            before = before_weights[name].cpu()
            after = param.data.cpu()
            total = before.numel()
            changed = (torch.abs(before - after) > eps).sum().item()
            percentage = 100 * changed / total
            if percentage>0:
              layer_names.append(name)
              changed_percentages.append(percentage)

    # Plot
    plt.figure(figsize=(6, max(6, len(layer_names) * 0.2)))
    plt.barh(layer_names, changed_percentages, color='orange')
    plt.xlabel("Percentuale di pesi modificati")
    plt.title("Variazione dei pesi per layer dopo il training")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------
# DATALOADER DEBUG: info about size and labels

def print_info_dataloader(dataloader):
    # reads all dataloader's labels
    labels = [l.item() for _, batch_labels in dataloader for l in batch_labels]
    print("Number of total examples:", len(labels))
    print("Number of distinct labels:", len(set(labels)))
    print("Distinct labels:", sorted(set(labels)))


def check_unique_images_in_dataloader(dataloader):
    """
    Checks whether all images in the dataloader are unique.
    Args:
        dataloader (DataLoader): The DataLoader to inspect.
    Returns:
        bool: True if all images are unique, False otherwise.
    """
    image_hashes = set()
    total = 0
    duplicates = 0

    for batch in dataloader:
        image = batch[0]  # Assumes (image, label)
        # Remove batch dimension if batch_size = 1
        image_tensor = image.squeeze().detach().cpu()

        # Flatten to 1D and convert to tuple to make it hashable
        image_hash = tuple(image_tensor.view(-1).tolist())

        total += 1
        if image_hash in image_hashes:
            duplicates += 1
        else:
            image_hashes.add(image_hash)

    print(f"Total images: {total}")
    print(f"Unique images: {len(image_hashes)}")
    print(f"Duplicates found: {duplicates}")

    return duplicates == 0


#-------------------------------------------------------------------------------
# DATALOADER SPLIT: stratification of the examples per label

def get_n_examples_per_class_loader(dataloader, num_classes=10, n_per_class=5):

    dataset = dataloader.dataset
    targets = dataset.targets if hasattr(dataset, 'targets') else [dataset[i][1] for i in range(len(dataset))]

    selected_indices = []
    class_counts = {}
    seen_classes = set()

    for idx, label in enumerate(targets):
        label = int(label)

        if label not in seen_classes:
            if len(seen_classes) >= num_classes:
                continue
            seen_classes.add(label)
            class_counts[label] = 0

        if class_counts[label] < n_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1

        # Early exit if done
        if all(class_counts.get(cls, 0) >= n_per_class for cls in seen_classes) and len(seen_classes) == num_classes:
            break

    subset = Subset(dataset, selected_indices)
    return DataLoader(subset, batch_size=1, shuffle=False)


#-------------------------------------------------------------------------------
# MASK COMPUTATION: functions for computing Fisher Scores and Mask

def compute_fisher_diagonal_per_example(model, dataloader, mask, soft_zero, device='cuda', num_examples=None):
    """
    Computes the Fisher diagonal on a model temporarily masked for the calculation.
    Input:
        model: The original, unpruned model.
        dataloader: The dataloader for data.
        mask: The binary mask to apply to the model's weights before computing scores.
        device: The device to run computations on.
        num_examples: The number of examples to use.
    Output:
        fisher_diag: The Fisher diagonal scores for the masked model.
    """
    # Store a copy of the original weights to restore them later
    original_weights = {name: p.clone().detach() for name, p in model.named_parameters()}

    # Apply the mask to the model's parameters for this computation
    with torch.no_grad():
      for name, p in model.named_parameters():
          if name in mask:
              current_mask = mask[name].to(device)
              current_mask[current_mask==0] = soft_zero
              p.mul_(current_mask)

    model.to(device)

    # Initialize Fisher diagonal
    fisher_diag = {
        name: torch.zeros_like(p, device='cpu')
        for name, p in model.named_parameters() if p.requires_grad
    }

    example_count = 0
    for batch_inputs, _ in dataloader:
        batch_size = batch_inputs.size(0)
        for i in range(batch_size):
            if num_examples is not None and example_count >= num_examples:
                break

            input_i = batch_inputs[i].unsqueeze(0).to(device)
            logits = model(input_i)
            log_probs = F.log_softmax(logits, dim=-1)
            distrib = torch.distributions.Categorical(logits=logits)
            y_sampled = distrib.sample()
            loss = F.nll_loss(log_probs, y_sampled, reduction='sum')

            model.zero_grad()
            loss.backward()

            for name, p in model.named_parameters():
                if p.requires_grad: # backbone weights
                    if p.grad is not None:
                        if name in mask:
                            fisher_diag[name] += (p.grad.detach().cpu() ** 2)
                            # fisher_diag[name][mask[name] == 0] = None # not necessary
                        else:
                            print(f"name: {name} was not in mask")

            example_count += 1

        if num_examples is not None and example_count >= num_examples:
            break

    # Restore the original weights to the model
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(original_weights[name])

    # Normalize
    if example_count > 0:
        for name in fisher_diag:
            fisher_diag[name] /= example_count

    return fisher_diag

def compute_mask_round(fisher_diag, round_target_sparsity, previous_mask, debug):
    """
    Computes a round mask by calculating a threshold ONLY on active weights.

    This function determines which of the currently active weights should be pruned
    in this specific round to meet the overall target sparsity for the round.

    Input:
        fisher_diag: A dictionary of Fisher scores for each parameter.
        round_target_sparsity: The cumulative sparsity target to be achieved by the end of this round.
        previous_mask: The cumulative mask from the previous round (r-1), where 1 means
                       the weight is active and 0 means it has already been pruned.
    Output:
        round_mask: A mask that prunes the new set of weights for this round. It will be
                    multiplied by the cumulative mask in the main loop.
    """
    # Step 1: Explicitly filter for active scores using the previous_mask
    # We will iterate through all layers to gather the scores of weights that have
    # not yet been pruned.
    active_scores = []
    total_params = 0
    num_already_pruned = 0

    # Iterate through each parameter (layer) in the model.
    for name in fisher_diag:
        # .view(-1) flattens the tensor to a 1D vector to make it easier to work with.
        scores = fisher_diag[name].view(-1)
        p_mask = previous_mask[name].view(-1)

        # This is the crucial filtering step. p_mask == 1 creates a boolean tensor
        # which is used to index scores. This selects ONLY the scores of weights
        # that were active (mask value of 1) in the previous round.
        active_scores.append(scores[p_mask == 1])

        # --- Update counters ---
        # Get the total number of parameters in this layer.
        total_params += len(p_mask)
        # Count how many weights in this layer were already pruned (mask value of 0).
        # .sum() on a boolean tensor counts the number of True elements.
        # .item() extracts the value from the resulting single-element tensor.
        num_already_pruned += (p_mask == 0).sum().item()

    # Concatenate the list of active score tensors from all layers into one large tensor.
    active_scores = torch.cat(active_scores)
    num_active_params = len(active_scores)

    # Step 2: Calculate the LOCAL sparsity needed for the active set
    # Based on the overall target for this round, calculate the total number of
    # weights that should be pruned cumulatively.
    num_to_prune_cumulatively = int(total_params * round_target_sparsity)

    # Calculate how many more weights we need to prune in this round to reach our target.
    num_to_prune_this_round = num_to_prune_cumulatively - num_already_pruned

    if debug:
        print(f"Round Target: {round_target_sparsity:.4f} ({num_to_prune_cumulatively} weights)")
        print(f"Total considered params: {total_params}")
        print(f"Already pruned: {num_already_pruned}. Active: {num_active_params}")
        print(f"Need to prune {num_to_prune_this_round} more weights from the active set.")


    # 1) Sort the 'active_scores' tensor in descending order.
    # The .values attribute is used to get only the sorted tensor, ignoring the indices.
    sorted_scores = torch.sort(active_scores, descending=True).values

    # 2) Select the value at the 'num_to_prune_this_round' index from the sorted tensor.
    # Note: 'num_to_prune_this_round' should be an integer representing the desired index.
    thr = sorted_scores[num_to_prune_this_round]

    # 'threshold' now holds the value from the 'quantile' position of the sorted scores.
    if debug:
        print(f"threshold:{thr}")

    round_mask = previous_mask #creazione alias, non copia
    for name in fisher_diag:
        bool_tensor = fisher_diag[name]>=thr
        round_mask[name][bool_tensor] = 0

    return round_mask

def compute_mask(model, dataloader, sparsity_target=0.9, R=5, soft_zero=0.1, num_examples=None, device='cuda', enable_plot=0, debug=False):
    """
    Computes the final pruning mask iteratively, pruning the most sensitive weights.

    Input:
        model: The neural network model to be pruned.
        dataloader: Dataloader for the data.
        sparsity_target: The final desired fraction of pruned weights.
        R: The number of rounds for iterative pruning.
        num_examples: The number of examples for each Fisher computation. None = use all the examples.
        device: The device to run computations on.
        enable_plot : plot the evolution of the sparsity-per-layer
    Output:
        final_mask: The final computed binary mask.
    """
    local_model = copy.deepcopy(model)
    # requires_grade True for all layers
    local_model.freeze(0)

    # requires_grade False for head, norm, token, embedding
    for name, param in local_model.named_parameters():
        if 'embed' in name or 'cls_token' in name or 'backbone.norm' in name or 'head' in name:
            param.requires_grad = False

    # Set model to evaluation mode
    local_model.eval()

    # print model structure
    if debug:
      local_model.debug()

    # --- Initialize the mask ---
    # We start with a mask of all ones, meaning no parameters are pruned initially.
    # torch.ones_like(p) creates a tensor of all ones with the same shape as p.
    final_mask = {
        name: torch.ones_like(p, device='cpu')
        for name, p in local_model.named_parameters() if p.requires_grad              # in the mask we consider the backbone layers only
    }

    # --- The iterative pruning loop ---
    # r must start from 1
    for r in range(1, R + 1):
        if debug:
            print(f"--- Starting Round {r}/{R} ---")

        # Compute Fisher scores using the cumulative mask from previous rounds.
        # This means we only calculate scores for weights that are still "alive".
            print("Computing Fisher diagonal with current mask...")
        fisher_diag = compute_fisher_diagonal_per_example(local_model, dataloader, final_mask, soft_zero,  device, num_examples)  # it's called final mask, just because at the end that variable will contain the final mask

        # increase the target sparsity at each round.
        current_sparsity = 1-(1-sparsity_target)**(r/R) # ex. (1-0.1^(r/R))
        if debug:
            print(f"Current sparsity level: {current_sparsity:.4f}")

        # Compute the new mask for this specific round based on the new sparsity target.
        final_mask = compute_mask_round(fisher_diag, current_sparsity, final_mask, debug)

        # debug: report the actual sparsity achieved (could be different from the imposed one)
        if debug:
            total_params = 0
            pruned_params = 0
            for name in final_mask:
                total_params += final_mask[name].numel()
                # (final_mask[name] == 0) creates a boolean tensor.
                # .sum() on a boolean tensor counts the number of True elements.
                # .item() extracts the value from a single-element tensor as a standard Python number.
                pruned_params += (final_mask[name] == 0).sum().item()
            actual_sparsity = pruned_params / total_params
            print(f"Achieved cumulative sparsity in final mask: {actual_sparsity:.4f}")
            print("-" * 25 + "\n")
        if(enable_plot==1):
            plot_all_layers_mask_sparsity(final_mask)

    return final_mask


#-------------------------------------------------------------------------------
# COMPUTE MASK: LOOP OVER THE CLIENTS WITH THE MODEL IN INPUT FOR THE CLIENT_MASK LIST COMPUTATION

def compute_mask_clients(model,
                         client_dataset,
                         num_classes,
                         n_per_class,
                         batch_size=128,
                         final_sparsity=0.9,
                         tot_rounds=10,
                         soft_zero=0.01,
                         num_examples=25,
                         debug = False):

  client_masks = [] 
  for i in range(len(client_dataset)):
    client_loader = DataLoader(client_dataset[i], batch_size=128, shuffle=True, num_workers=2)

    if debug:
      print_info_dataloader(client_loader)

    stratified_loader = get_n_examples_per_class_loader(client_loader, num_classes=num_classes, n_per_class=n_per_class) # num_classes * n_per_class to be constant
    if debug:
      print_info_dataloader(stratified_loader)

    start = time.time()
    mask = compute_mask(model, stratified_loader, sparsity_target=final_sparsity, R=tot_rounds, num_examples=num_examples, soft_zero=soft_zero, device='cuda', enable_plot=0, debug=False) # num_examples=None : sfoglia tutto il dataset passato in ingresso (già ridotto)
    end = time.time()
    
    if debug:
      plot_all_layers_mask_sparsity(mask)
      plot_qkv_weight_bias_sparsity(mask)
      print(f"Mask computation time: {end-start}")
      print(f"Sparsity target: {final_sparsity}")
      print(f"Soft Zero Value: {soft_zero}")
      print(f"Rounds: {tot_rounds}")
      print(f"Num_examples: {num_examples}")

    client_masks.append(mask)

  return client_masks

#-------------------------------------------------------------------------------
# CONVERT FLOAT MASK TO A BOOLEAN ONE (SAVING MEMORY)
def convert_float_masks_to_bool(masks_list):
    """
    Convert a list of float masks (dicts) to boolean masks.

    Args:
        masks_list (list of dict): Each dict maps name -> float tensor (with 0.0/1.0).

    Returns:
        list of dict: Each dict maps name -> bool tensor.
    """
    bool_masks = []
    for client_mask in masks_list:
        bool_mask = {name: p.bool() for name, p in client_mask.items()}
        bool_masks.append(bool_mask)
    return bool_masks

#-------------------------------------------------------------------------------
# COMPARE LISTS OF MASKS 
def compare_mask_lists(float_masks_list: list, bool_masks_list: list):
    """
    Confronta due liste di maschere float e bool.

    Args:
        float_masks_list: lista di dict {name: tensor(float)}
        bool_masks_list: lista di dict {name: tensor(bool)}

    Stampa quali maschere non combaciano per ogni elemento della lista.
    """
    if len(float_masks_list) != len(bool_masks_list):
        print("⚠️ Attenzione: le due liste hanno lunghezze diverse!")
        return

    for idx, (float_mask, bool_mask) in enumerate(zip(float_masks_list, bool_masks_list)):
        mismatches = []

        for name in float_mask:
            if name not in bool_mask:
                print(f"[Elemento {idx}] Parametro '{name}' non trovato nella maschera booleana.")
                continue

            f_tensor = float_mask[name]
            b_tensor = bool_mask[name]

            if not torch.equal(f_tensor.bool(), b_tensor):
                mismatches.append(name)

        if not mismatches:
            print(f"[Elemento {idx}] ✅ Tutti i parametri combaciano!")
        else:
            print(f"[Elemento {idx}] ⚠️ Mismatch in {len(mismatches)} parametri: {mismatches}")

