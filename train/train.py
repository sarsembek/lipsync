import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lipsync_model import LipSyncModel
from train.dataset import LipSyncDataset
from utils.device import get_device

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Get data and move to device
        mfcc = batch['mfcc'].to(device)
        landmarks = batch['landmarks'].to(device)
        teacher_input = batch['teacher_input'].to(device)
        
        # Forward pass with teacher forcing
        optimizer.zero_grad()
        outputs = model(mfcc, teacher_input)
        
        # Calculate loss
        loss = criterion(outputs, landmarks)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * mfcc.size(0)
    
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data and move to device
            mfcc = batch['mfcc'].to(device)
            landmarks = batch['landmarks'].to(device)
            
            # Forward pass (no teacher forcing during validation)
            outputs = model(mfcc)
            
            # Calculate loss
            loss = criterion(outputs, landmarks)
            total_loss += loss.item() * mfcc.size(0)
    
    return total_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Train LipSync model')
    parser.add_argument('--audio_dir', type=str, default='data/lrs2_rebuild/audio_process', 
                        help='Directory containing MFCC features')
    parser.add_argument('--landmark_dir', type=str, default='data/lrs2_rebuild/landmark', 
                        help='Directory containing facial landmarks')
    parser.add_argument('--train_file', type=str, default='data/lrs2_rebuild/train.txt', 
                        help='File containing training filenames')
    parser.add_argument('--val_file', type=str, default='data/lrs2_rebuild/dev.txt', 
                        help='File containing validation filenames')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=50, 
                        help='Sequence length (0 for dynamic)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, 
                        help='Ratio of teacher forcing (0-1)')
    parser.add_argument('--hidden_dim', type=int, default=64, 
                        help='Hidden dimension size')
    parser.add_argument('--resume', type=str, default='', 
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = LipSyncDataset(
        args.audio_dir, 
        args.landmark_dir, 
        args.train_file, 
        sequence_length=args.sequence_length
    )
    
    # Try to load validation dataset if the file exists
    val_dataset = None
    try:
        val_dataset = LipSyncDataset(
            args.audio_dir, 
            args.landmark_dir, 
            args.val_file, 
            sequence_length=args.sequence_length
        )
        print(f"✅ Found {len(val_dataset)} validation samples")
    except Exception as e:
        print(f"⚠️ Error loading validation data: {e}")
        print("⚠️ Continuing without validation.")
        
        # If no validation dataset, use a portion of the training dataset
        if len(train_dataset) > 0:
            # Split 90% for training, 10% for validation
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            print(f"Using {val_size} samples from train set as validation")
    
    print(f"Using sequence length: {args.sequence_length}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    
    # Initialize model
    model = LipSyncModel(
        input_dim=13,  # MFCC features
        face_dim=40,   # Landmark features (20 points x 2 coordinates)
        hidden_dim=args.hidden_dim,
        output_dim=40  # 20 points x 2 coordinates
    ).to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Optionally use a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = float('inf')
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"lipsync_checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)
        
        # Print info
        time_taken = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - Time: {time_taken:.1f}s - Train Loss: {train_loss:.4f}", end="")
        if val_loader is not None:
            print(f" - Val Loss: {val_loss:.4f}")
        else:
            print()
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, "lipsync_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
