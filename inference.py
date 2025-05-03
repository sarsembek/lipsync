import torch
import cv2
import dlib
import librosa
import numpy as np
import os
import argparse
from models.lipsync_model import LipSyncModel
from preprocessing.audio_features import extract_mfcc
from utils.device import get_device
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Set up face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Loaded face landmark predictor")
except RuntimeError:
    print("❌ Error: Could not load the shape_predictor_68_face_landmarks.dat file.")
    print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit(1)

def extract_mouth_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    # Use first detected face
    face = faces[0]
    
    # Get facial landmarks
    landmarks = predictor(gray, face)
    
    # Extract mouth landmarks (indices 48-67)
    mouth_landmarks = []
    for i in range(48, 68):
        point = landmarks.part(i)
        mouth_landmarks.append((point.x, point.y))
    
    return np.array(mouth_landmarks)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    orig_landmarks = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        landmarks = extract_mouth_landmarks(frame)
        
        if landmarks is not None:
            orig_landmarks.append(landmarks.flatten())
        else:
            # If no face detected, use the previous landmarks or zeros
            if orig_landmarks:
                orig_landmarks.append(orig_landmarks[-1])
            else:
                orig_landmarks.append(np.zeros(40))
    
    cap.release()
    return frames, np.array(orig_landmarks), fps

def visualize_landmarks(frame, landmarks, color=(0, 255, 0)):
    """Draw landmarks on a frame"""
    lm_array = landmarks.reshape(-1, 2)
    
    # Draw points
    img = frame.copy()
    for (x, y) in lm_array:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
    
    # Draw lines to connect the mouth landmarks
    for i in range(len(lm_array) - 1):
        pt1 = (int(lm_array[i][0]), int(lm_array[i][1]))
        pt2 = (int(lm_array[i + 1][0]), int(lm_array[i + 1][1]))
        cv2.line(img, pt1, pt2, color, 1)
    
    # Connect the last point with the first
    pt1 = (int(lm_array[-1][0]), int(lm_array[-1][1]))
    pt2 = (int(lm_array[0][0]), int(lm_array[0][1]))
    cv2.line(img, pt1, pt2, color, 1)
    
    return img

def create_comparison_animation(frames, original_landmarks, predicted_landmarks, fps, output_path):
    """Create animation showing original vs predicted landmarks"""
    num_frames = min(len(frames), len(original_landmarks), len(predicted_landmarks))
    
    # Default to GIF for safety, since we're having issues with ffmpeg
    output_path = os.path.splitext(output_path)[0] + '.gif'
    print(f"Using GIF format for animation: {output_path}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame_idx):
        ax.clear()
        
        # Get the current frame
        frame = frames[frame_idx]
        orig_lm = original_landmarks[frame_idx].reshape(-1, 2)
        pred_lm = predicted_landmarks[frame_idx].reshape(-1, 2)
        
        # Display the video frame
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Plot original landmarks (green)
        ax.plot(orig_lm[:, 0], orig_lm[:, 1], 'go-', label='Original', linewidth=1, markersize=3)
        
        # Plot predicted landmarks (red)
        ax.plot(pred_lm[:, 0], pred_lm[:, 1], 'ro-', label='Predicted', linewidth=1, markersize=3)
        
        ax.set_title(f'Frame {frame_idx}')
        ax.legend()
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    anim = FuncAnimation(
        fig, update, frames=range(num_frames), interval=1000/fps
    )
    
    try:
        # Use pillow writer for gif which is almost always available
        print("Saving animation with pillow writer...")
        # Use a lower frame rate for GIFs to keep the file size reasonable
        gif_fps = min(fps, 10)
        anim.save(output_path, writer='pillow', fps=gif_fps, dpi=100)
        print(f"✅ Saved visualization to {output_path}")
    except Exception as e:
        print(f"⚠️ Error saving animation: {e}")
        print("⚠️ Animation could not be saved. You might need to install additional dependencies.")
    
    plt.close(fig)

def create_video_with_landmarks(frames, original_landmarks, predicted_landmarks, fps, output_path):
    """Create video with side-by-side comparison of original and predicted landmarks"""
    num_frames = min(len(frames), len(original_landmarks), len(predicted_landmarks))
    
    # Get the dimensions of the video
    height, width, _ = frames[0].shape
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    for i in range(num_frames):
        frame = frames[i]
        orig_lm = original_landmarks[i]
        pred_lm = predicted_landmarks[i]
        
        # Draw landmarks on the frames
        orig_frame = visualize_landmarks(frame, orig_lm, color=(0, 255, 0))  # Green for original
        pred_frame = visualize_landmarks(frame, pred_lm, color=(0, 0, 255))  # Red for predicted
        
        # Add text labels
        cv2.putText(orig_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pred_frame, "Predicted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Combine frames
        combined_frame = np.hstack((orig_frame, pred_frame))
        
        # Write frame
        out.write(combined_frame)
    
    out.release()
    print(f"✅ Saved video to {output_path}")

def create_lip_sync_video(frames, predicted_landmarks, audio_path, fps, output_path):
    """Create a video with the predicted lip movements and the input audio"""
    num_frames = min(len(frames), len(predicted_landmarks))
    
    # Get the dimensions of the video
    height, width, _ = frames[0].shape
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # Create frames with predicted landmarks
    for i in range(num_frames):
        frame = frames[i]
        pred_lm = predicted_landmarks[i]
        
        # Draw predicted landmarks on the frame
        frame_with_landmarks = visualize_landmarks(frame, pred_lm, color=(0, 0, 255))  # Red for predicted
        
        # Write frame
        out.write(frame_with_landmarks)
    
    out.release()
    print(f"✅ Saved temporary video to {temp_video_path}")
    
    # Try to create the final video with audio
    print("Adding audio to video...")
    
    # Create a simple function to make a new video with audio
    def create_video_with_audio(video_path, audio_path, output_path):
        """Create a new video file with audio using OpenCV and librosa"""
        import shutil
        import subprocess
        
        try:
            # First try using ffmpeg directly if it's available
            if shutil.which("ffmpeg"):
                cmd = f"ffmpeg -y -i {video_path} -i {audio_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest {output_path}"
                subprocess.run(cmd, shell=True, check=True)
                return True
        except Exception as e:
            print(f"⚠️ Error with ffmpeg: {e}")
            
        # If ffmpeg fails or is not available, try a pure Python approach
        try:
            # Use a direct pip install for moviepy if needed
            try:
                import moviepy.editor as mp
            except ImportError:
                print("Installing moviepy...")
                subprocess.run([sys.executable, "-m", "pip", "install", "moviepy"], check=True)
                import moviepy.editor as mp
                
            # Load video and audio
            video_clip = mp.VideoFileClip(video_path)
            audio_clip = mp.AudioFileClip(audio_path)
            
            # If audio is longer than video, trim it
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            
            # Set audio
            final_clip = video_clip.set_audio(audio_clip)
            
            # Write output file
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            # Close clips
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            return True
        except Exception as e:
            print(f"⚠️ Error with moviepy: {e}")
            return False
                
    # Try to add audio to the video
    import sys
    import shutil
    success = create_video_with_audio(temp_video_path, audio_path, output_path)
    
    if success:
        print(f"✅ Successfully added audio to video: {output_path}")
        # Remove temporary video file
        os.remove(temp_video_path)
    else:
        print(f"❌ Unable to add audio automatically.")
        print(f"The video without audio is available at: {temp_video_path}")
        print(f"You can manually combine the video and audio using:")
        print(f"  1. Install ffmpeg and run: ffmpeg -i {temp_video_path} -i {audio_path} -c:v copy -c:a aac -shortest {output_path}")
        print(f"  2. Or use a video editor of your choice")
        # Copy the temp file to the output path
        shutil.copy(temp_video_path, output_path)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Test LipSync model on input video and audio')
    parser.add_argument('--model_path', type=str, default='checkpoints/lipsync_final.pth', 
                        help='Path to the trained model checkpoint')
    parser.add_argument('--video_path', type=str, default='input/video.mp4', 
                        help='Path to the input video')
    parser.add_argument('--audio_path', type=str, default='input/audio.wav', 
                        help='Path to the input audio')
    parser.add_argument('--output_path', type=str, default='output', 
                        help='Directory to save output files')
    parser.add_argument('--sequence_length', type=int, default=50, 
                        help='Sequence length for inference')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # 1) Load the trained model
    device = get_device()
    print(f"Using device: {device}")
    
    model = LipSyncModel().to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(f"✅ Loaded model from {args.model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)
    
    # 2) Process video to get frames and original landmarks
    print("Processing video...")
    frames, orig_landmarks, fps = process_video(args.video_path)
    print(f"✅ Processed {len(frames)} frames")
    
    # 3) Process audio to get MFCC features
    print("Processing audio...")
    try:
        mfcc = extract_mfcc(args.audio_path)
        print(f"✅ Extracted MFCC features: {mfcc.shape}")
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        exit(1)
    
    # 4) Run inference
    print("Running inference...")
    # Ensure same sequence length (this might require padding or truncation)
    seq_len = args.sequence_length
    min_len = min(len(frames), mfcc.shape[0])
    
    if min_len < seq_len:
        print(f"⚠️ Warning: Available data length ({min_len}) is less than the sequence length ({seq_len}).")
        print(f"Using dynamic sequence length: {min_len}")
        seq_len = min_len
    
    # Prepare input for the model (truncate if needed to match sequence length)
    if mfcc.shape[0] > seq_len:
        # Center-crop the MFCC features
        start = (mfcc.shape[0] - seq_len) // 2
        mfcc_input = mfcc[start:start + seq_len]
    else:
        # Pad with zeros if shorter than the desired sequence length
        pad_before = (seq_len - mfcc.shape[0]) // 2
        pad_after = seq_len - mfcc.shape[0] - pad_before
        mfcc_input = np.pad(mfcc, ((pad_before, pad_after), (0, 0)), mode='constant')
    
    # Convert to torch tensor
    mfcc_tensor = torch.tensor(mfcc_input, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        predicted_landmarks = model(mfcc_tensor)
        
    # Convert predictions to numpy
    predicted_landmarks = predicted_landmarks.squeeze(0).cpu().numpy()
    
    # Stretch predicted landmarks to match the original video length
    if len(frames) != seq_len:
        # Interpolate to match the video frame count
        import scipy.interpolate as interp
        
        # Create time vectors for original and target sequences
        pred_times = np.linspace(0, 1, seq_len)
        target_times = np.linspace(0, 1, len(frames))
        
        # Create interpolation function for each landmark coordinate
        interp_func = interp.interp1d(pred_times, predicted_landmarks, axis=0)
        
        # Interpolate to get values at target_times
        predicted_landmarks = interp_func(target_times)
    
    # 5) Create visual comparison - both animated and as video
    print("Creating visualizations...")
    # First create the video comparison (this should work as it uses OpenCV)
    animation_path = os.path.join(args.output_path, 'landmark_comparison.mp4')
    create_video_with_landmarks(frames, orig_landmarks, predicted_landmarks, fps, animation_path)
    
    # Then create the matplotlib animation
    matplotlib_path = os.path.join(args.output_path, 'landmark_animation.gif')  # Use .gif instead of .mp4
    create_comparison_animation(frames, orig_landmarks, predicted_landmarks, fps, matplotlib_path)
    
    # Create lip-synced video with input audio
    lip_sync_video_path = os.path.join(args.output_path, 'lip_sync_video.mp4')
    create_lip_sync_video(frames, predicted_landmarks, args.audio_path, fps, lip_sync_video_path)
    
    print(f"✅ All results saved to {args.output_path}")

if __name__ == '__main__':
    main()