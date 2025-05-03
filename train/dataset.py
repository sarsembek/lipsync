import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LipSyncDataset(Dataset):
    def __init__(self, audio_dir, landmark_dir, filenames_file, sequence_length=50):
        self.audio_dir = audio_dir
        self.landmark_dir = landmark_dir
        self.sequence_length = sequence_length
        
        # Load filenames
        with open(filenames_file, 'r') as f:
            self.filenames = [line.strip() for line in f]
        
        # Filter out unavailable files
        self.available_samples = []
        self.check_availability()
        
    def check_availability(self):
        """Check if both MFCC and landmark files exist for each sample"""
        missing_count = 0
        for filename in self.filenames:
            mfcc_file = os.path.join(self.audio_dir, f"{filename}.npy")
            landmark_file = os.path.join(self.landmark_dir, f"{filename}.npy")
            
            mfcc_exists = os.path.exists(mfcc_file)
            landmark_exists = os.path.exists(landmark_file)
            
            if not (mfcc_exists and landmark_exists):
                missing_count += 1
                print(f"⚠️  Missing for `{filename}`: mfcc={mfcc_exists}, landmark={landmark_exists}")
                continue
                
            self.available_samples.append(filename)
        
        print(f"⚠️  Warning: {missing_count}/{len(self.filenames)} entries missing ({missing_count/len(self.filenames)*100:.1f}%)! Check your train.txt vs folders.")
        if len(self.available_samples) > 0:
            print(f"✅ Continuing with {len(self.available_samples)} available files.")
        else:
            raise RuntimeError("No valid training samples found!")
    
    def __len__(self):
        return len(self.available_samples)
    
    def __getitem__(self, idx):
        filename = self.available_samples[idx]
        
        # Load MFCC features and landmarks
        mfcc_file = os.path.join(self.audio_dir, f"{filename}.npy")
        landmark_file = os.path.join(self.landmark_dir, f"{filename}.npy")
        
        mfcc = np.load(mfcc_file)
        landmarks = np.load(landmark_file)
        
        # Ensure consistent sequence length
        if self.sequence_length > 0:
            # Fixed sequence length mode
            seq_len = self.sequence_length
            
            # Adjust sequence length if either input is too short
            if mfcc.shape[0] < seq_len or landmarks.shape[0] < seq_len:
                seq_len = min(mfcc.shape[0], landmarks.shape[0])
            
            # Center crop or pad mfcc
            if mfcc.shape[0] > seq_len:
                start = (mfcc.shape[0] - seq_len) // 2
                mfcc = mfcc[start:start + seq_len]
            else:
                # Pad with zeros
                pad_before = (seq_len - mfcc.shape[0]) // 2
                pad_after = seq_len - mfcc.shape[0] - pad_before
                mfcc = np.pad(mfcc, ((pad_before, pad_after), (0, 0)), mode='constant')
            
            # Center crop or pad landmarks
            if landmarks.shape[0] > seq_len:
                start = (landmarks.shape[0] - seq_len) // 2
                landmarks = landmarks[start:start + seq_len]
            else:
                # Pad with zeros
                pad_before = (seq_len - landmarks.shape[0]) // 2
                pad_after = seq_len - landmarks.shape[0] - pad_before
                landmarks = np.pad(landmarks, ((pad_before, pad_after), (0, 0)), mode='constant')
        else:
            # Dynamic sequence length mode - use the shorter sequence length
            seq_len = min(mfcc.shape[0], landmarks.shape[0])
            
            # Center crop both
            if mfcc.shape[0] > seq_len:
                start = (mfcc.shape[0] - seq_len) // 2
                mfcc = mfcc[start:start + seq_len]
            
            if landmarks.shape[0] > seq_len:
                start = (landmarks.shape[0] - seq_len) // 2
                landmarks = landmarks[start:start + seq_len]
        
        # Prepare the data for the modified model with teacher forcing
        # Create shifted landmarks for teacher forcing (previous frame input)
        # The first frame's input will be zeros, and we'll use landmarks[:-1] for the rest
        teacher_input = np.zeros_like(landmarks)
        teacher_input[1:] = landmarks[:-1]  # Shift by one frame
        
        return {
            'mfcc': torch.tensor(mfcc, dtype=torch.float32),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'teacher_input': torch.tensor(teacher_input, dtype=torch.float32)
        }
