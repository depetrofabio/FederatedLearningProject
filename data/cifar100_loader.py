import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np

# CIFAR 100 contains 100 calsses. There are 500 images per class in the train set and 100 per class in the test set.
# For a total of 600 images per class. So it has 50000 training examples and 10000 test examples.
# The 100 classes (fine label) are actually grouped in 20 super-classes (coarse label) => every image has 2 labels: class,superclass.

def get_cifar100(train=True, transform=None): # https://dev.to/hyperkai/cifar100-in-pytorch-4p8d
    ''' This function returns the train dataset CIFAR100 (50000 images).'''
    return CIFAR100(root='./data', train=train, download=True, transform=transform) # NOTE: if download=True the entire (train + test) dataset is downloaded locally

def split_train_val(dataset, val_ratio=0.2):
    '''This function: TRAIN -> train + val'''
    return random_split(dataset, [1-val_ratio, val_ratio]) # returns [train,val]

def create_non_iid_splits(dataset, num_clients=100, classes_per_client= 10): # as dataset use the train (after the split train,val)
    ''' Assign data to clients with Dirichlet distribution (simulate label skew). ATT! it only assigns different set of labels to different clients'''
    num_classes = 100
    # classes_per_client = num_classes/num_clients <-- TODO

    if isinstance(dataset, Subset):
        # Extract targets from the original dataset using Subset indices
        labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        # If the data are not a dataset we can access more direclty. While Subsets do not have the target attribute.
        labels = np.array(dataset.targets)
    
    client_data = {i: [] for i in range(num_clients)}   # create a dictionary where keys goes from 0 to num_clients-1, values are empty lists.
    for cls in range(num_classes):
        indices = np.where(labels == cls)[0]
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients // classes_per_client) # Divides the indices into equal parts, where each part corresponds to a subset of data for clients. ( // is integer division (flor division)) 
        for idx, split in enumerate(splits): # Assign each split to a client 
            client_id = (cls % classes_per_client) + (idx * classes_per_client)
            client_data[client_id].extend(split.tolist())
    return [Subset(dataset, indices) for indices in client_data.values()] #  the function returns a list of Subset objects, each representing the data assigned to a client