# Funzione per eseguire il training di un'epoca
import torch
import wandb
import itertools
from FederatedLearningProject.checkpoints.checkpointing import save_checkpoint


def train_epoch(model, train_loader, optimizer, criterion, device):

    train_loss = 0.0
    correct_train = 0
    total_train = 0

    model.train() # the model train() method should have been overrided if some blocks must be in evaluation mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler step in train_and_validate()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    avg_train_loss = train_loss / total_train


    return avg_train_loss, train_accuracy

# Funzione per eseguire la validazione
def validate_epoch(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / total
    val_accuracy = 100 * correct / total

    return avg_val_loss, val_accuracy


# Funzione per eseguire il tes
def test_epoch(model, test_loader, criterion, device):
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / total
    test_accuracy = 100 * correct / total

    return avg_test_loss, test_accuracy



# Funzione per il logging su W&B
def log_to_wandb(epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "epoch": epoch
    }, step=epoch)

def log_to_wandb_test(epoch, train_loss, train_accuracy, test_loss, test_accuracy):
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "epoch": epoch
    }, step=epoch)

# Funzione principale per l'allenamento e la validazione
def train_and_validate(start_epoch, model, train_loader, val_loader, scheduler, optimizer, criterion, device, checkpoint_path, num_epochs, checkpoint_interval):
    #start_epoch = 1

    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        train_loss, train_accuracy = train_epoch(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, device=device)

        # Validation
        val_loss, val_accuracy = validate_epoch(model=model, val_loader=val_loader, criterion=criterion, device=device)

        # Logging su W&B
        log_to_wandb(epoch, train_loss=train_loss, train_accuracy=train_accuracy, val_loss=val_loss, val_accuracy=val_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)

        # Scheduler step and debug
        scheduler.step()
        # Output
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}")
        print(f"current LR: {scheduler.get_last_lr()}")

        # Salvataggio dei checkpoint
        if epoch % checkpoint_interval == 0:
            save_checkpoint(epoch, model, optimizer,  scheduler, train_loss, val_loss, checkpoint_path)

    # wandb.finish()
    print(f"[train and validate]: final val accuracy: {val_accuracy}")
    return val_accuracy


# Funzione principale per l'allenamento e il test
def train_and_test(start_epoch, model, train_loader, test_loader, scheduler, optimizer, criterion, device, checkpoint_path, num_epochs, checkpoint_interval):
    #start_epoch = 1

    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        train_loss, train_accuracy = train_epoch(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, device=device)

        # Test
        test_loss, test_accuracy = test_epoch(model=model, test_loader=test_loader, criterion=criterion, device=device)

        # Logging su W&B
        log_to_wandb_test(epoch, train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)

        # Scheduler step and debug
        scheduler.step()
        # Output
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"current LR: {scheduler.get_last_lr()}")

        # Salvataggio dei checkpoint
        if epoch % checkpoint_interval == 0:
            save_checkpoint_test(epoch, model, optimizer,  scheduler, train_loss, test_loss, checkpoint_path)

    # wandb.finish()
    print(f"[train and Test]: final test accuracy: {test_accuracy}")
    return test_accuracy

def generate_configs(config):
    '''
    INPUT:  hyperparameters dictionary with all values of each param.
    OUTPUT: all possible configurations with a corresponding key.

    Example of input variable (config):
    {
    lr: {'values': [0.003]}
    weight: {'values': [0.0001, 0.0005]}
    }

    Example of returned variable (configs):
    {
     0: {'lr': 0.003, 'weight_decay': 0.0001},
     1: {'lr': 0.003, 'weight_decay': 0.0005}
     }

    '''

    # Separa i parametri variabili e fissi
    grid_params = {k: v['values'] for k, v in config.items() if 'values' in v}

    # Tutte le combinazioni possibili dei parametri variabili
    keys, values = zip(*grid_params.items())    # take keys and values separately
    # Save in a list of dictionaries all combinations, by using a cartesian product.
    # Each dictionary represent a unique combination
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Genera dizionario numerato
    configs = {}
    for idx, combo in enumerate(combinations): # note combo ~ unique combination in a dict
        configs[idx] = combo
    return configs