import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import Subset, DataLoader, random_split, dataset
import numpy as np
import matplotlib.pyplot as plt
# from google.colab import drive

# CIFAR 100 contains 100 calsses. There are 500 images per class in the train set and 100 per class in the test set.
# For a total of 600 images per class. So it has 50000 training examples and 10000 test examples.
# The 100 classes (fine label) are actually grouped in 20 super-classes (coarse label) => every image has 2 labels: class,superclass.


# the professor suggested to treat the validation as similar as possible the test set
# this implies that we have to only apply the normalization step on them



class TransformedSubset(torch.utils.data.Dataset):
    """Wrapper dataset that applies a transform to a subset"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_cifar100(valid_split_perc=0.2, seed=42):
    """
    Load CIFAR-100 dataset with train/val/test splits and appropriate transforms
    
    Args:
        valid_split_perc: Percentage of training data to use for validation (0.0-1.0)
        seed: Random seed for reproducible splits
        
    Returns:
        trainset, valset, testset: Dataset objects with appropriate transforms
    """
    # Normalization parameters
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]

    # Base transforms (applied to validation and test)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Training transforms (augmentation + base)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load full training set (without transforms initially)
    full_train = CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=None
    )

    # Create splits
    train_size = int((1 - valid_split_perc) * len(full_train))
    val_size = len(full_train) - train_size
    
    # Split with reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train, 
        [train_size, val_size],
        generator=generator
    )

    # Apply transforms using our wrapper class
    trainset = TransformedSubset(train_subset, train_transform)
    valset = TransformedSubset(val_subset, base_transform)

    # Load test set with base transforms
    testset = CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=base_transform
    )

    # Print statistics
    print(f"Number of images in Training Set: {len(trainset)}")
    print(f"Number of images in Validation Set: {len(valset)}")
    print(f"Number of images in Test Set: {len(testset)}")
    print("\n✅ Datasets loaded successfully")
    
    return trainset, valset, testset

def create_iid_splits(dataset, num_clients = 100):
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users

      
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

def plot_same_class(train_images, train_labels, test_images, test_labels, class_id, num_samples=5):
    # Filtra immagini del training set con la classe specificata
    train_class_mask = np.array(train_labels) == class_id
    train_class_images = train_images[train_class_mask]
    
    # Filtra immagini del test set con la stessa classe
    test_class_mask = np.array(test_labels) == class_id
    test_class_images = test_images[test_class_mask]
    
    # Prendi un campione casuale
    train_samples = train_class_images[np.random.choice(len(train_class_images), num_samples, replace=False)]
    test_samples = test_class_images[np.random.choice(len(test_class_images), num_samples, replace=False)]
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Classe: {classes[class_id]}', fontsize=16)
    
    # Plot training set
    plt.subplot(1, 2, 1)
    plt.title("Training Set")
    plt.imshow(np.hstack(train_samples))
    plt.axis('off')
    
    # Plot test set
    plt.subplot(1, 2, 2)
    plt.title("Test Set")
    plt.imshow(np.hstack(test_samples))
    plt.axis('off')
    
    plt.show()