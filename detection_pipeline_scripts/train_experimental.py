# Imports
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from mobilenet_model import WatermarkDetectionMobileNetV3
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define function to evaluate model on validation data and return accuracy and loss
def evaluate(model, loader, criterion, device):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    total_loss = 0.0
    correct = 0
    total = 0

    # Disable gradients
    with torch.no_grad():
        # Batch through dataset
        for images, labels in loader:
            # Move to device
            images, labels = images.to(device), labels.to(device)
            
            # Foward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Add loss to total loss
            total_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Count the number of correct predictions
            correct += (preds == labels).sum().item()
            
            # Track total number of samples
            total += labels.size(0)
    
    # Calculate average loss for batch
    avg_loss = total_loss / len(loader)
    
    # Calculate accuracy
    accuracy = correct / total
    
    return avg_loss, accuracy

# Define function to train watermark detection model on experimental dataset
def train_experimental(train_dir, val_dir, model_save_path, epochs, batch_size, lr):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print device being used
    print(f"Experimental training using device: {device}")

    # Define image preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to expected input size
        transforms.ToTensor() # Convert to tensor
    ])

    # Load training and validation datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize pretrained model
    model = WatermarkDetectionMobileNetV3(pretrained=True)
    
    # Move to device
    model.to(device)

    # Define loss
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initliaze lists to store losses and accuracies 
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # Initialize best validation loss comparison
    best_val_loss = float("inf")
    
    # Initialize counter for early stopping
    epochs_no_improve = 0
    
    # Define patience for early stopping
    early_stop_patience = 10
    
    # Define scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Go through each epoch
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        
        # Initialize variables
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Go through batch
        for images, labels in train_loader:
            # Move to device
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backpropagation
            loss.backward()
            
            # Optimizer step
            optimizer.step()

            # Add loss to running loss
            running_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Count the number of correct predictions
            correct += (preds == labels).sum().item()
            
            # Track total number of samples
            total += labels.size(0)

        # Calculae training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Run evaluation on validation set and get validation metrics
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print summary for the epoch
        print(f"[Experimental][Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save model if validation loss decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping occured!")  # Print status update
                break

    # Make sure directory exists and save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save()
    torch.save(model.state_dict(), model_save_path)
    
    # Print where model is saved
    print(f"Experimental model saved at: {model_save_path}")

    # Plot loss convergence
    epochs_range = range(1, len(train_losses)+1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Convergence (Experimental)")
    plt.legend()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/loss_convergence_experimental.png")

    # Plot accuracy convergence
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accs, label="Train Accuracy")
    plt.plot(epochs_range, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Convergence (Experimental)")
    plt.legend()
    plt.savefig("results/accuracy_convergence_experimental.png")
    