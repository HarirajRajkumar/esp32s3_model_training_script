#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Script for ESP32-S3 Motherboard Detection

This script trains a model for motherboard detection using either:
1. A custom CNN architecture, or
2. A pre-trained MobileNetV2 model (recommended)

The trained model is saved in PyTorch format and exported to ONNX.
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2

from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class MotherboardClassifier(nn.Module):
    """Simple CNN for motherboard classification."""
    
    def __init__(self, num_classes=2):
        super(MotherboardClassifier, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Adaptive pooling to handle different input sizes
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_data_loaders(data_dir, batch_size=32, img_size=224):
    """Create data loaders for training and validation."""
    
    # Define data transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def create_mobilenet_model(num_classes=2, pretrained=True):
    """Create a MobileNetV2 model with custom classifier."""
    
    # Load pre-trained model
    model = mobilenet_v2(pretrained=pretrained)
    
    # Modify the classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, 
               scheduler, device, num_epochs=25, save_dir='models/pytorch'):
    """Train the model."""
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        # Process batches
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        # Process validation batches
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass (no gradient calculation needed)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Update statistics
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
        
        # Calculate validation statistics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        # Print statistics
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Saved new best model")
        
        print()
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    return model, history


def plot_training_history(history, save_dir):
    """Plot and save training and validation metrics."""
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on the test set."""
    
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    
    # Storage for detailed analysis
    all_preds = []
    all_labels = []
    
    # Evaluate batches
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        
        # Update statistics
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
        
        # Store predictions and labels for detailed analysis
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate test statistics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc, all_preds, all_labels


def export_to_onnx(model, sample_input, output_path, input_names=['input'], output_names=['output']):
    """Export PyTorch model to ONNX format."""
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export the model
    torch.onnx.export(
        model,                       # Model being exported
        sample_input,                # Example input tensor
        output_path,                 # Output file path
        export_params=True,          # Export model parameters
        opset_version=11,            # ONNX version
        do_constant_folding=True,    # Optimize constants
        input_names=input_names,     # Input tensor names
        output_names=output_names,   # Output tensor names
        dynamic_axes={'input': {0: 'batch_size'},  # Variable batch size
                     'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to {output_path}")


def main():
    """Main function to parse arguments and run model training."""
    
    parser = argparse.ArgumentParser(description="ESP32-S3 Motherboard Detection Model Training")
    
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Base directory for dataset")
    
    parser.add_argument("--model_type", type=str, default="mobilenet", choices=["custom", "mobilenet"],
                       help="Type of model to train (custom CNN or MobileNetV2)")
    
    parser.add_argument("--pretrained", action="store_true",
                       help="Use pretrained weights for MobileNetV2")
    
    parser.add_argument("--img_size", type=int, default=224,
                       help="Input image size")
    
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    
    parser.add_argument("--num_epochs", type=int, default=15,
                       help="Number of training epochs")
    
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Initial learning rate")
    
    parser.add_argument("--save_dir", type=str, default="models",
                       help="Directory to save trained models")
    
    args = parser.parse_args()
    
    # Set up paths
    pytorch_save_dir = os.path.join(args.save_dir, "pytorch")
    onnx_save_dir = os.path.join(args.save_dir, "onnx")
    
    os.makedirs(pytorch_save_dir, exist_ok=True)
    os.makedirs(onnx_save_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
        args.data_dir, batch_size=args.batch_size, img_size=args.img_size
    )
    
    print(f"Classes: {class_to_idx}")
    
    # Save class mapping
    with open(os.path.join(args.save_dir, 'class_mapping.txt'), 'w') as f:
        for class_name, class_id in class_to_idx.items():
            f.write(f"{class_id}: {class_name}\n")
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    if args.model_type == "custom":
        model = MotherboardClassifier(num_classes=len(class_to_idx))
        print("Using custom CNN architecture")
    else:  # mobilenet
        model = create_mobilenet_model(num_classes=len(class_to_idx), pretrained=args.pretrained)
        print(f"Using MobileNetV2 {'with' if args.pretrained else 'without'} pretrained weights")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    
    if args.model_type == "custom":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:  # For MobileNetV2, use a smaller learning rate
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate if not args.pretrained else args.learning_rate/10)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    print("Starting training...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=args.num_epochs, save_dir=pytorch_save_dir
    )
    
    # Evaluate model on test set
    print("Evaluating on test set...")
    test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # Export model to ONNX
    print("Exporting model to ONNX format...")
    # Create a sample input
    sample_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    onnx_path = os.path.join(onnx_save_dir, "motherboard_detector.onnx")
    export_to_onnx(model, sample_input, onnx_path)
    
    print("Training completed successfully!")
    print(f"PyTorch model saved to: {os.path.join(pytorch_save_dir, 'best_model.pth')}")
    print(f"ONNX model saved to: {onnx_path}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
