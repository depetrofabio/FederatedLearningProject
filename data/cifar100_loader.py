import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import Subset, DataLoader, random_split, Dataset
import numpy as np
import matplotlib.pyplot as plt
import heapq
from google.colab import drive
import warnings

# CIFAR 100 contains 100 calsses. There are 500 images per class in the train set and 100 per class in the test set.
# For a total of 600 images per class. So it has 50000 training examples and 10000 test examples.
# The 100 classes (fine label) are actually grouped in 20 super-classes (coarse label) => every image has 2 labels: class,superclass.


# The professor suggested to treat the validation as similar as possible the test set
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


def get_cifar100(valid_split_perc: float = 0.2,
                 seed: int = 42,
                 transf: bool = True):
    """
    Load CIFAR-100 with train/val/test splits and apply transforms via TransformedSubset.
    Returns either (trainset, valset, testset) with transforms, or the raw Subsets if transf=False.

    Returns: train, val, test
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load raw datasets with no transforms
    full_train = CIFAR100(root='./data', train=True,  download=True, transform=None)
    test_raw   = CIFAR100(root='./data', train=False, download=True, transform=None)

    # Split train into train/val
    train_size = int((1 - valid_split_perc) * len(full_train))
    val_size   = len(full_train) - train_size
    gen        = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(full_train, [train_size, val_size], generator=gen)

    # Wrap in TransformedSubset
    trainset = TransformedSubset(train_sub, train_transform)
    valset   = TransformedSubset(val_sub,   base_transform)
    testset  = TransformedSubset(test_raw,  base_transform)

    print(f"Number of images in Training Set:   {len(trainset)}")
    print(f"Number of images in Validation Set: {len(valset)}")
    print(f"Number of images in Test Set:       {len(testset)}")
    print("✅ Datasets loaded successfully")

    if transf:
        return trainset, valset, testset
    else:
        return train_sub, val_sub, test_raw


def create_iid_splits(dataset: Dataset, num_clients: int = 100, keep_transformations = True, debug=True):
    # """
    # Split dataset IID into `num_clients` equal portions.
    # Returns dict: client_id -> set of indices.
    # """
    # N = len(dataset)
    # num_items = N // num_clients
    # all_idxs = np.arange(N)
    # dict_users = {}

    # for i in range(num_clients):
    #     chosen = np.random.choice(all_idxs, num_items, replace=False)
    #     dict_users[i] = set(chosen)
    #     all_idxs = np.setdiff1d(all_idxs, chosen, assume_unique=True)

    if isinstance(dataset, TransformedSubset):
        base = dataset.subset
    elif isinstance(dataset, Subset):
        base = dataset
    else:
        base = None

    if base is not None:
        orig_indices = np.array(base.indices)
        targets = np.array(base.dataset.targets)[base.indices]
    else:
        orig_indices = np.arange(len(dataset))
        targets = np.array(dataset.targets)

    unique_classes = np.unique(targets) # Create an array of unique labels
    num_classes = len(unique_classes)   # Number of classes

    list_of_subsets = create_non_iid_splits(dataset=dataset, num_clients=num_clients, classes_per_client=num_classes, debug=debug, keep_transformations=keep_transformations)

    return list_of_subsets


def create_non_iid_splits(dataset: Dataset,  
                          num_clients: int = 100,
                          classes_per_client: int = 10,
                          random_state: int = 42,
                          keep_transformations: bool = True,
                          debug: bool = True):
    """
    Create non-IID splits by assigning to each client shards of data
    from only classes_per_client classes.
    Works with Dataset, Subset, or TransformedSubset.

    Returns: list of Subset objects (one per client).
    """
    # Unwrap Subset/TransformedSubset to get base indices & targets
    # .indices -> get the original indices from the dataset
    # .targets -> get the labels
    if isinstance(dataset, TransformedSubset):
        base = dataset.subset
    elif isinstance(dataset, Subset):
        base = dataset
    else:
        base = None

    if base is not None:
        orig_indices = np.array(base.indices)
        targets = np.array(base.dataset.targets)[base.indices]
    else:
        orig_indices = np.arange(len(dataset))
        targets = np.array(dataset.targets)

    unique_classes = np.unique(targets) # Create an array of unique labels
    num_classes = len(unique_classes)   # Number of classes

    # Chek if the input values make sense
    if classes_per_client > num_classes:
        raise ValueError("classes_per_client cannot exceed number of classes")
    if num_clients * classes_per_client < num_classes:
        raise ValueError("num_clients * classes_per_client is too low, some classes would not be utilised ")
    if (num_clients * classes_per_client) % num_classes !=0:
        print(f"ATT! num_clients * classes_per_client is not a multiple of num_classes. \n>> This scenario is not implemented yet.\n>> num_clients: {num_clients} = v, num_classes: {num_classes} = c, classes_per_client: {classes_per_client} = r\n>> For this function to work properly we need (v*r) % c = 0\n>> s = (v*r)/c  <--> s*c = v*r , where s is the number of shards we need to split each class into.\n>> With this approach s must be an integer.\nNo actions performed")
        return None
    if classes_per_client*num_clients > len(orig_indices):
        print(f"The number of necessary shards is greater than the number of samples \nNo actions performed")
        return None

    if debug:
        print(f"Dataset has {len(targets)} samples across {num_classes} classes.")
        print(f"Creating {num_clients} {'IID' if num_classes == classes_per_client else 'non IID'} splits with {classes_per_client} classes each.\n")


    # Build shards per class
    indices_by_label = {lbl: orig_indices[targets == lbl] for lbl in unique_classes} # Each class is associated with the indices of its samples 
    shards_per_class = (num_clients * classes_per_client) // num_classes             # Calculate the number of shards each class must be spli into

    # Check if the split is possible in a safe way (np.array_split() will work anyway but the result is unpredictable)
    if min([len(indices_by_label[lbl]) for lbl in indices_by_label.keys()])<shards_per_class:
        print(f"There is at least a class with not enough samples to perform a safe split into {shards_per_class} shards.")
        user_choice = input("Do you want to continue? (y/n): ").lower()
        if user_choice != 'y':
            print('No actions performed')
            return None
        
    # Create a dictionary where classes are associated with a list of shards
    class_partitions = {
        lbl: np.array_split(indices_by_label[lbl], shards_per_class)
        for lbl in unique_classes
    }

    if debug:
        print(f"\nEach of the {num_classes} classes split into {shards_per_class} shards.")

    availability = {lbl: shards_per_class for lbl in unique_classes} # initialize the shards availability count for each class
    clients_data_indices = {}                                        # this dictionary will contain the indices of the samples, divided by client

    # NOTE: np.array_split Behavior: 
    # When you use np.array_split(array, N), and the length of the array is not perfectly divisible by N, NumPy tries to make the shards as equal in size as possible. 
    # It does this by making the first len(array) % N shards one element larger than the remaining shards.
    # Since the shards are assigned from the last to the first for each class, the last clients would receive more samples of all classes.
    # For this reason we shuffle the shards before the assignement.
    rng_shuffle = np.random.default_rng(random_state)
    for lbl in unique_classes:
        rng_shuffle.shuffle(class_partitions[lbl])

    # Assign shards to each client
    # Strategy: we take the necessary shard_per_class number of shards, for the given client, from the classes with higher availability
    for c_id in range(num_clients): # iterate over clients
        # pick classes with most remaining shards.
        # heapq.nlargest will return the availability.items which value part is the higher - as specified in the lambda function.
        top = heapq.nlargest(classes_per_client, availability.items(), key=lambda x: x[1])
        # Initialize the container for the shards of the current client
        picks = [] 
        for lbl, avail in top: # iterate over the (class_labels, availability) pairs
            if avail <= 0:
                raise RuntimeError(f"Client {c_id} wants a shard from class {lbl} -- No shards left for class {lbl}")
            # Compute shard index. Ex. l=[shard0, shard1, shard3] avail=3 --> take l[2] = shard3 for the current client
            shard_idx = avail - 1
            picks.append(class_partitions[lbl][shard_idx]) # To access the indices of the shard we need to access the dictionary [lbl] and index the list [shard_idx]
            availability[lbl] -= 1 # update availability
        
        # The full set of examples for the client is now ordered by class, therefore we shuffle the examples.
        arr = np.concatenate(picks)
        rng = np.random.default_rng(random_state + c_id) # the shuffle is deterministic knowing the random state. The shuffle will be different for each client
        rng.shuffle(arr)
        clients_data_indices[c_id] = arr    # Assign the shuffled indices to the current client (c_id)
    
    if debug:
        print(f'\nChecking unique classes that each client sees:')
        idx_to_lbl = {idx: lbl for lbl, indices in indices_by_label.items() for idx in indices}
        client_unique_classes = [(i, set([idx_to_lbl[idx] for idx in clients_data_indices[i]])) for i in range(num_clients)]
        for item in client_unique_classes:
            print(f"Client {item[0]} has samples from classes: {item[1]}")
            xx = len(item[1])
            if xx!=classes_per_client:
                print(f'SOMETHING IS WRONG')
            print(f"Total: {xx}")

        print(f"\n")


    # Build final Subsets
    client_datasets = []

    if keep_transformations:
        # map original -> local
        if base is not None:
            index_map = {orig: loc for loc, orig in enumerate(base.indices)}    # build a map from the root indices to the local indices
        else:
            index_map = {i: i for i in orig_indices}

        for c_id, orig_idxs in clients_data_indices.items():
            local_idxs = [index_map[i] for i in orig_idxs]      # map dataset indices to the subset indices
            client_datasets.append(Subset(dataset, local_idxs)) # use the local indices to directly accede to the dataset passed as function argument
            if debug:
                print(f"Client {c_id}: {len(local_idxs)} samples")
    else:
        # use raw dataset as root
        root = base.dataset if base is not None else dataset
        for c_id, orig_idxs in clients_data_indices.items():
            client_datasets.append(Subset(root, orig_idxs))
            if debug:
                print(f"Client {c_id}: {len(orig_idxs)} samples")
    
    if debug:
        if keep_transformations:
            print(f"Client partitions created from {dataset}, passed as an argument to the function. All the transformations were mantained.")
        else:
            print(f"Client partitions created from the root_dataset. No transformations applied to data were mantained.")

    return client_datasets