# Funzione per eseguire il training di un'epoca
import torch
import wandb

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.backbone.eval()       # fix backbone stats
    model.classifier.train()    # enable head training
    
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    avg_train_loss = train_loss / total_train
    
    return avg_train_loss, train_accuracy

# Funzione per eseguire la validazione
def validate_epoch(model, val_loader, criterion, device):
    model.backbone.eval()      # still frozen
    model.classifier.eval()    # disable head dropout, if any
    
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

# Funzione per il salvataggio del checkpoint
def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, checkpoint_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, checkpoint_path)
    print(f"Checkpoint salvato su: {checkpoint_path}")

# Funzione per il logging su W&B
def log_to_wandb(epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "epoch": epoch
    }, step=epoch)

# Funzione principale per l'allenamento e la validazione
def train_and_validate(start_epoch, model, train_loader, val_loader, optimizer, criterion, device,  checkpoint_path, num_epochs, checkpoint_interval):
    #start_epoch = 1

    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)

        # Logging su W&B
        log_to_wandb(epoch, train_loss, train_accuracy, val_loss, val_accuracy)

        # Output
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Salvataggio dei checkpoint
        if epoch % checkpoint_interval == 0:
            save_checkpoint(epoch, model, optimizer, train_loss, val_loss, checkpoint_path)

    wandb.finish()