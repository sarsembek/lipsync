import torch
from torch.utils.data import DataLoader
import os
import argparse
from models.lipsync_model import LipSyncModel
from train.dataset import LipSyncDataset
from train.train import train
from utils.device import get_device
from utils.io import load_file_list

if __name__ == "__main__":
    # Command line arguments for flexible training configuration
    parser = argparse.ArgumentParser(description='Train LipSync model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_list', type=str, default="data/lrs2_rebuild/train.txt", help='Path to training list')
    parser.add_argument('--val_list', type=str, default="data/lrs2_rebuild/dev.txt", help='Path to validation list')
    parser.add_argument('--audio_dir', type=str, default="data/lrs2_rebuild/audio_process", help='Directory containing audio features')
    parser.add_argument('--landmark_dir', type=str, default="data/lrs2_rebuild/landmark", help='Directory containing landmarks')
    parser.add_argument('--save_dir', type=str, default="checkpoints", help='Directory to save models')
    parser.add_argument('--strict', action='store_true', help='Strictly require all files to be present')
    parser.add_argument('--sequence_length', type=int, default=50, help='Target sequence length for training (use 0 for dynamic length)')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Set up device
    device = get_device()
    print(f"Using device: {device}")

    # 2) Load training data
    mfcc_files, landmark_files = load_file_list(args.train_list, args.audio_dir, args.landmark_dir, strict=args.strict)
    print(f"✅ Found {len(mfcc_files)} training samples")
    
    # Only proceed if we have at least some training data
    if len(mfcc_files) == 0:
        print("❌ Error: No training samples found. Please check your data paths.")
        exit(1)
    
    # 3) Load validation data if available
    val_mfcc_files, val_landmark_files = [], []
    if os.path.exists(args.val_list):
        try:
            val_mfcc_files, val_landmark_files = load_file_list(args.val_list, args.audio_dir, args.landmark_dir, strict=False)
            print(f"✅ Found {len(val_mfcc_files)} validation samples")
        except Exception as e:
            print(f"⚠️ Error loading validation data: {e}")
            print("⚠️ Continuing without validation.")
    else:
        print("⚠️ No validation file found. Training without validation.")

    # Use dynamic length if sequence_length is 0
    seq_len = args.sequence_length if args.sequence_length > 0 else None
    print(f"Using sequence length: {'Dynamic' if seq_len is None else seq_len}")
    
    # 4) Create model and datasets
    model = LipSyncModel().to(device)
    train_dataset = LipSyncDataset(mfcc_files, landmark_files, sequence_length=seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataloader = None
    if val_mfcc_files and val_landmark_files:
        val_dataset = LipSyncDataset(val_mfcc_files, val_landmark_files, sequence_length=seq_len)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5) Train the model
    best_model = train(
        model=model, 
        dataloader=train_dataloader, 
        device=device,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )

    # 6) Save the final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'lipsync_final.pth'))
    print(f"Training completed. Final model saved to {os.path.join(args.save_dir, 'lipsync_final.pth')}")
