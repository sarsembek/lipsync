#!/usr/bin/env python3
import os
import glob
import cv2
import dlib
import numpy as np
from tqdm import tqdm
import argparse
import librosa
from preprocessing.audio_features import extract_mfcc

# Set up face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Loaded face landmark predictor")
except RuntimeError:
    print("❌ Error: Could not load the shape_predictor_68_face_landmarks.dat file.")
    print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("Extract it and place it in the project root directory.")
    exit(1)

# Create directories if they don't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def extract_mouth_landmarks(frame, face_rect):
    # Get facial landmarks
    landmarks = predictor(frame, face_rect)
    
    # Extract mouth landmarks (indices 48-67)
    mouth_landmarks = []
    for i in range(48, 68):
        point = landmarks.part(i)
        mouth_landmarks.append((point.x, point.y))
    
    return np.array(mouth_landmarks)

def process_video_file(video_path, landmark_output_dir, base_name):
    """Process a video file to extract mouth landmarks"""
    landmark_file = os.path.join(landmark_output_dir, f"{base_name}.npy")
    
    # Skip if landmark file already exists
    if os.path.isfile(landmark_file):
        return 0
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 1
    
    all_landmarks = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        # If a face is detected, get mouth landmarks
        if len(faces) > 0:
            face = faces[0]  # Use the first face detected
            try:
                mouth_landmarks = extract_mouth_landmarks(gray, face)
                # Flatten the landmarks for storage
                all_landmarks.append(mouth_landmarks.flatten())
            except:
                # If landmark detection fails, use the previous frame's landmarks or zeros
                if all_landmarks:
                    all_landmarks.append(all_landmarks[-1])
                else:
                    all_landmarks.append(np.zeros(40))  # 20 points with x,y coordinates
        else:
            # No face detected, use the previous frame's landmarks or zeros
            if all_landmarks:
                all_landmarks.append(all_landmarks[-1])
            else:
                all_landmarks.append(np.zeros(40))
    
    cap.release()
    
    # Save landmarks as numpy array
    if all_landmarks:
        np.save(landmark_file, np.array(all_landmarks))
        return 0
    else:
        return 1

def process_audio_file(audio_path, mfcc_output_dir, base_name):
    """Process an audio file to extract MFCC features"""
    mfcc_file = os.path.join(mfcc_output_dir, f"{base_name}.npy")
    
    # Skip if MFCC file already exists
    if os.path.isfile(mfcc_file):
        return 0
    
    try:
        # Extract MFCC features
        mfcc = extract_mfcc(audio_path)
        # Save MFCC features as numpy array
        np.save(mfcc_file, mfcc)
        return 0
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return 1

def process_dataset(args):
    """Process the entire dataset"""
    # Create output directories
    ensure_dir(args.mfcc_output_dir)
    ensure_dir(args.landmark_output_dir)
    
    # Get list of IDs from train.txt/test.txt/dev.txt
    ids = []
    for list_file in [args.train_list, args.val_list, args.test_list]:
        if os.path.isfile(list_file):
            with open(list_file, "r") as f:
                ids.extend([line.strip() for line in f.readlines()])
    
    print(f"Found {len(ids)} sample IDs in list files")
    
    # Initialize counters
    total_videos = len(ids)
    processed_videos = 0
    skipped_videos = 0
    failed_videos = 0
    processed_audio = 0
    skipped_audio = 0
    failed_audio = 0
    
    # Process each sample
    for sample_id in tqdm(ids, desc="Processing samples"):
        # Look for video file
        video_path = os.path.join(args.video_dir, f"{sample_id}.mp4")
        
        # Look for audio files (could be in different subdirectories)
        audio_paths = glob.glob(os.path.join(args.audio_dir, "**", f"{sample_id}*.wav"), recursive=True)
        
        # Process video file if it exists
        if os.path.isfile(video_path):
            result = process_video_file(video_path, args.landmark_output_dir, sample_id)
            if result == 0:
                processed_videos += 1
            else:
                failed_videos += 1
        else:
            skipped_videos += 1
        
        # Process audio file if it exists
        if audio_paths:
            audio_path = audio_paths[0]  # Use the first matching audio file
            result = process_audio_file(audio_path, args.mfcc_output_dir, sample_id)
            if result == 0:
                processed_audio += 1
            else:
                failed_audio += 1
        else:
            # Try using mix files from min/tr/mix directories
            audio_paths = glob.glob(os.path.join(args.audio_dir, "min", "**", "mix", "*.wav"), recursive=True)
            matching_paths = [p for p in audio_paths if sample_id in os.path.basename(p)]
            
            if matching_paths:
                audio_path = matching_paths[0]
                result = process_audio_file(audio_path, args.mfcc_output_dir, sample_id)
                if result == 0:
                    processed_audio += 1
                else:
                    failed_audio += 1
            else:
                skipped_audio += 1
    
    # Print summary
    print("\n===== Processing Summary =====")
    print(f"Total samples: {total_videos}")
    print(f"Videos: Processed {processed_videos}, Failed {failed_videos}, Skipped {skipped_videos}")
    print(f"Audio: Processed {processed_audio}, Failed {failed_audio}, Skipped {skipped_audio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess video and audio files for the lipsync model')
    parser.add_argument('--video_dir', type=str, default='data/lrs2_rebuild/faces', help='Directory containing video files')
    parser.add_argument('--audio_dir', type=str, default='data/lrs2_rebuild/audio/wav16k', help='Directory containing audio files')
    parser.add_argument('--mfcc_output_dir', type=str, default='data/lrs2_rebuild/audio_process', help='Directory to save MFCC features')
    parser.add_argument('--landmark_output_dir', type=str, default='data/lrs2_rebuild/landmark', help='Directory to save landmarks')
    parser.add_argument('--train_list', type=str, default='data/lrs2_rebuild/train.txt', help='Path to training sample list')
    parser.add_argument('--val_list', type=str, default='data/lrs2_rebuild/dev.txt', help='Path to validation sample list')
    parser.add_argument('--test_list', type=str, default='data/lrs2_rebuild/test.txt', help='Path to test sample list')
    
    args = parser.parse_args()
    process_dataset(args)