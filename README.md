"""
Simple Training Script for PPE Detection
Perfect for beginners to understand the basics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
import time

# ============================================================================
# STEP 1: CONFIGURATION (Change these if needed)
# ============================================================================

DATASET_PATH = "../Dataset_Sample"           # Path to dataset
IMAGES_PATH = f"{DATASET_PATH}/images/train" # Training images
LABELS_PATH = f"{DATASET_PATH}/labels/train" # Training labels
MODEL_SAVE_PATH = "models/ppe_model.pth"    # Where to save trained model
NUM_EPOCHS = 10                              # Number of training cycles
BATCH_SIZE = 4                               # Number of images per batch
LEARNING_RATE = 0.001                        # How fast model learns
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# STEP 2: DATASET CLASS
# ============================================================================

class PPEDataset(Dataset):
    """
    Custom Dataset for PPE Detection
    
    This class:
    1. Loads images from disk
    2. Loads annotations (labels)
    3. Prepares data for training
    """
    
    def __init__(self, images_dir, labels_dir, img_size=416):
        """
        Initialize dataset
        
        Args:
            images_dir: Folder containing images
            labels_dir: Folder containing labels
            img_size: Size to resize images to
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        
        # Get all image files
        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_files:
            self.image_files = sorted(self.images_dir.glob("*.png"))
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        """Return total number of images"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get one image and its label
        
        Returns:
            image: Preprocessed image tensor
            label: Object labels for the image
        """
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            return None, None
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 range
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor (PyTorch format)
        # Note: PyTorch uses (Channels, Height, Width)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Dummy label (replace with actual labels from annotation files)
        # In real scenario, read from .txt files
        label = torch.tensor([0, 0, 0, 0, 0])  # 5 PPE classes
        
        return image, label

# ============================================================================
# STEP 3: SIMPLE MODEL ARCHITECTURE
# ============================================================================

class SimplePPEModel(nn.Module):
    """
    Simple CNN model for PPE Detection
    
    Architecture:
    - Input: 3 channels (RGB) x 416x416 pixels
    - Convolutional layers: Extract features
    - Dense layers: Classification
    - Output: Probabilities for each PPE class
    """
    
    def __init__(self, num_classes=5):
        """
        Initialize model layers
        
        Args:
            num_classes: Number of PPE types to detect
        """
        super(SimplePPEModel, self).__init__()
        
        # Convolutional layers (feature extraction)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)      # 416x416 -> 416x416
        self.pool = nn.MaxPool2d(2, 2)                                # Reduce size by half
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)     # 208x208 -> 208x208
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)    # 104x104 -> 104x104
        
        # Dense layers (classification)
        # 128 filters * (416/8)^2 pixels
        self.fc1 = nn.Linear(128 * 52 * 52, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through network
        
        Args:
            x: Input images (batch_size, 3, 416, 416)
        
        Returns:
            Output predictions (batch_size, num_classes)
        """
        # Convolutional block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 208x208
        
        # Convolutional block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 104x104
        
        # Convolutional block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # 52x52
        
        # Flatten for dense layers
        x = x.view(x.size(0), -1)  # Flatten to 1D
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============================================================================
# STEP 4: TRAINING FUNCTION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Steps:
    1. Forward pass (predict)
    2. Calculate loss (error)
    3. Backward pass (calculate gradients)
    4. Update weights (optimize)
    
    Args:
        model: The neural network model
        dataloader: Data for training
        criterion: Loss function
        optimizer: Optimization algorithm
        device: CPU or GPU
    
    Returns:
        Average loss for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Skip if data loading failed
        if images is None:
            continue
        
        # Move data to device (CPU/GPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # ---- STEP 1: Forward Pass ----
        # Model makes predictions
        outputs = model(images)
        
        # ---- STEP 2: Calculate Loss ----
        # Loss = How wrong the model was
        loss = criterion(outputs, labels)
        
        # ---- STEP 3: Backward Pass ----
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Calculate gradients
        loss.backward()
        
        # ---- STEP 4: Update Weights ----
        # Update model parameters
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss

# ============================================================================
# STEP 5: MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function"""
    
    print("="*60)
    print("🎯 PPE Detection Model Training")
    print("="*60)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = PPEDataset(IMAGES_PATH, LABELS_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} images")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = SimplePPEModel(num_classes=5)
    model = model.to(DEVICE)
    print(f"Model moved to {DEVICE}")
    
    # Loss function (measures how wrong the model is)
    # BCEWithLogitsLoss is good for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer (algorithm to update weights)
    # Adam is a popular choice - learns well
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler (reduce learning rate over time)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training history
    training_losses = []
    
    # ---- START TRAINING ----
    print("\n" + "="*60)
    print("🚀 Starting training...")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📈 Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-"*60)
        
        # Train for one epoch
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, DEVICE)
        training_losses.append(avg_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        elapsed_time = time.time() - start_time
        print(f"\n✓ Epoch {epoch + 1} complete!")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Elapsed Time: {elapsed_time/60:.1f} minutes")
    
    # ---- TRAINING COMPLETE ----
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Final loss: {training_losses[-1]:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to {MODEL_SAVE_PATH}...")
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("✓ Model saved successfully!")
    
    # Print tips
    print("\n" + "="*60)
    print("📌 Next Steps:")
    print("="*60)
    print("1. Run detection on images: python detect.py --image image.jpg")
    print("2. Test on video: python detect.py --video video.mp4")
    print("3. Improve model by adding more training data")
    print("4. Tune hyperparameters (learning rate, epochs, etc.)")
    print("="*60)

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()
