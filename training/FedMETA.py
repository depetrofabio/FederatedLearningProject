import torch
import torch.nn as nn
import copy
from typing import List, Dict

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