import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plotting Function
def plot_all_layers_mask_sparsity(final_mask):
    layer_names = []
    active_percentages = []

    for name, mask in final_mask.items():
        total_params = mask.numel()
        active_params = (mask == 1).sum().item()
        percentage = 100 * active_params / total_params

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
    
    
        

def compute_fisher_diagonal_per_example(model, dataloader, mask, device='cuda', num_examples=None, soft_zero = 0.001):
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

    # Apply the provided mask to the model's parameters for this computation
    # with torch.no_grad():
    with torch.no_grad():
      for name, p in model.named_parameters():
          if name in mask:
              current_mask = mask[name].to(device)
              current_mask[current_mask==0] = soft_zero
              p.mul_(current_mask)   #TODO 0 morbido

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
                if p.requires_grad:
                    if p.grad is not None:
                        # Only accumulate gradients for weights that have not been pruned
                        if name in mask:
                            fisher_diag[name] += (p.grad.detach().cpu() ** 2)
                            # fisher_diag[name][mask[name] == 0] = None
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

def compute_mask_round(fisher_diag, round_target_sparsity, previous_mask, debug=True):
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
    # --- Step 1: Explicitly filter for active scores using the previous_mask ---
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

    # --- Step 2: Calculate the LOCAL sparsity needed for the active set ---
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
    print(f"threshold:{thr}")

    round_mask = previous_mask
    for name in fisher_diag:
        bool_tensor = fisher_diag[name]>=thr
        round_mask[name][bool_tensor] = 0

    return round_mask

def compute_mask(model, dataloader, sparsity_target=0.9, R=5, num_examples=None, device='cuda', enable_plot=0):
    """
    Computes the final pruning mask iteratively, pruning the most sensitive weights.

    Input:
        model: The neural network model to be pruned.
        dataloader: Dataloader for the data.
        sparsity_target: The final desired fraction of pruned weights.
        R: The number of rounds for iterative pruning.
        num_examples: The number of examples for each Fisher computation. None = use all the examples.
        device: The device to run computations on.
    Output:
        final_mask: The final computed binary mask.
    """
    # requires_grade True for all layers
    model.freeze(0)

    # requires_grade False for head, norm, token, embedding
    for name, param in model.named_parameters():
        if 'embed' in name or 'cls_token' in name or 'backbone.norm' in name or 'head' in name:
            param.requires_grad = False

    # Set model to evaluation mode
    model.eval()

    # --- Initialize the mask ---
    # We start with a mask of all ones, meaning no parameters are pruned initially.
    # torch.ones_like(p) creates a tensor of all ones with the same shape as p.
    final_mask = {
        name: torch.ones_like(p, device='cpu')
        for name, p in model.named_parameters() if p.requires_grad
    }

    # --- The iterative pruning loop ---
    # r must start from 1
    for r in range(1, R + 1):
        print(f"--- Starting Round {r}/{R} ---")

        # Compute Fisher scores using the cumulative mask from previous rounds.
        # This means we only calculate scores for weights that are still "alive".
        print("Computing Fisher diagonal with current mask...")
        fisher_diag = compute_fisher_diagonal_per_example(model, dataloader, final_mask, device, num_examples)  # it's called final mask, just because at the end that variable will contain the final mask

        # increase the target sparsity at each round.
        current_sparsity = 1-(1-sparsity_target)**(r/R) # ex. (1-0.1^(r/R))
        print(f"Current sparsity level: {current_sparsity:.4f}")

        # Compute the new mask for this specific round based on the new sparsity target.
        final_mask = compute_mask_round(fisher_diag, current_sparsity, final_mask)

        # --- Report the actual sparsity achieved ---
        total_params = 0
        pruned_params = 0
        for name in final_mask:
            # .numel() returns the total number of elements in a tensor.
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