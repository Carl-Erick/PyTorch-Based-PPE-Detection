"""
Training module for PPE Detection Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Callable
import os
from tqdm import tqdm
import json
from datetime import datetime


class Trainer:
    """
    Trainer class for PPE detection model training and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Initialize Trainer.
        
        Args:
            model: PyTorch model for training
            device: Device to train on (cuda/cpu)
            num_classes: Number of PPE classes
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization)
        """
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # Handle different target formats
            if isinstance(targets, dict):
                labels = targets['labels'].to(self.device)
            else:
                labels = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            if isinstance(targets, dict):
                # For detection models
                logits = self.model(images)[0]
            else:
                # For classification models
                logits = self.model(images)
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = logits.argmax(dim=1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / labels.size(0)
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")
            for batch_idx, (images, targets) in enumerate(progress_bar):
                images = images.to(self.device)
                
                # Handle different target formats
                if isinstance(targets, dict):
                    labels = targets['labels'].to(self.device)
                else:
                    labels = targets.to(self.device)
                
                # Forward pass
                if isinstance(targets, dict):
                    logits = self.model(images)[0]
                else:
                    logits = self.model(images)
                
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions = logits.argmax(dim=1)
                correct = (predictions == labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        save_dir: str = './checkpoints',
        save_best_only: bool = True
    ):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_best_only: Only save best checkpoint
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            self._save_checkpoint(checkpoint_path, epoch + 1, val_loss)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                self._save_checkpoint(best_checkpoint_path, epoch + 1, val_loss)
                print(f"✓ Best model saved to {best_checkpoint_path}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
        print(f"{'='*60}")
    
    def _save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
    
    def save_history(self, filepath: str):
        """Save training history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {filepath}")
