import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np
from google.colab import drive

# CIFAR 100 contains 100 calsses. There are 500 images per class in the train set and 100 per class in the test set.
# For a total of 600 images per class. So it has 50000 training examples and 10000 test examples.
# The 100 classes (fine label) are actually grouped in 20 super-classes (coarse label) => every image has 2 labels: class,superclass.

def get_cifar100(train=True, download=True): # https://dev.to/hyperkai/cifar100-in-pytorch-4p8d
    ''' This function returns the train dataset CIFAR100 (50000 images). Already transformed to be given in input to the model'''
    if train:
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),  # image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # Normalize to match the pretraining
        ])
    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return CIFAR100(root='./data', train=train, download=download, transform=transform) # NOTE: if download=True the entire (train + test) dataset is downloaded locally

def prova():
    print("ciao")

def split_train_val(dataset, val_ratio=0.2):
    '''This function: TRAIN -> train + val'''
    return random_split(dataset, [1-val_ratio, val_ratio]) # returns [train,val]

def create_non_iid_splits(dataset, num_clients=100, classes_per_client= 10, dirichlet = False): # as dataset use the train (after the split train,val)
    ''' Create non iid splits'''
    if isinstance(dataset, Subset):
        # Extract targets from the original dataset using Subset indices
        labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        # If the data are not a dataset we can access more direclty. While Subsets do not have the target attribute.
        labels = np.array(dataset.targets)

    num_classes = len(np.unique(labels))
    print(f"\n debug num_classes = {num_classes} \n")

    client_data = {i: [] for i in range(num_clients)}   # create a dictionary where keys goes from 0 to num_clients-1, values are empty lists.

    if dirichlet:
        alpha = 1.5 # controls the distribution skew
        for cls in range(num_classes):
            indices = np.where(labels == cls)[0]
            np.random.shuffle(indices)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (proportions * len(indices)).astype(int)
            proportions[-1] = len(indices) - np.sum(proportions[:-1])  # Adjust last client
            splits = np.split(indices, proportions.cumsum())[:-1]
            for client_id, split in enumerate(splits):
                client_data[client_id].extend(split.tolist())
    else:
        for cls in range(num_classes):
            indices = np.where(labels == cls)[0]
            np.random.shuffle(indices)
            splits = np.array_split(indices, num_clients // classes_per_client) # Divides the indices into equal parts, where each part corresponds to a subset of data for clients. ( // is integer division (flor division)) 
            for idx, split in enumerate(splits): # Assign each split to a client 
                client_id = (cls % classes_per_client) + (idx * classes_per_client)
                client_data[client_id].extend(split.tolist())

    return [Subset(dataset, indices) for indices in client_data.values()] #  the function returns a list of Subset objects, each representing the data assigned to a client