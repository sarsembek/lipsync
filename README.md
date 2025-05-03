# Lipsync CNN LSTM
## 1. Project Structure
```bash
lipsync-cnn-lstm/
│
├── data/               # Audio and video dataset
├── preprocessing/      # Scripts to extract audio features and facial landmarks
├── models/             # Your CNN-LSTM model definition
├── train/              # Training and validation logic
├── inference/          # For generating lip-synced mouth landmarks or videos
├── utils/              # Misc utilities (e.g., for visualization, metrics)
├── requirements.txt
└── main.py             # Entry point
```
## 2. Install Required Packages

Requirements
```bash
torch
torchaudio
opencv-python
dlib
librosa
scikit-learn
matplotlib
numpy
moviepy
```
Install with
```bash
pip install -r requirements.txt
```

# lipsync
