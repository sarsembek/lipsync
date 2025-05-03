import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        x, _ = self.lstm(x)
        return x  # (batch, seq_len, hidden_dim*2) due to bidirectional LSTM

class FaceEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.lstm = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x_reshaped = x.reshape(-1, x.size(2))
        
        x = self.fc1(x_reshaped)
        x = x.reshape(batch_size, seq_len, -1)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(x)
        
        x_reshaped = x.reshape(-1, x.size(2))
        x = self.fc2(x_reshaped)
        x = x.reshape(batch_size, seq_len, -1)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(x)
        
        x, _ = self.lstm(x)
        return x  # (batch, seq_len, hidden_dim*2)

class FaceDecoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=40):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x, _ = self.lstm(x)
        
        batch_size, seq_len, _ = x.shape
        x_reshaped = x.reshape(-1, x.size(2))
        
        x = self.fc1(x_reshaped)
        x = x.reshape(batch_size, seq_len, -1)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(x)
        
        x = self.fc2(x.reshape(-1, x.size(2)))
        x = x.reshape(batch_size, seq_len, -1)
        return x  # (batch, seq_len, output_dim)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim*2, hidden_dim)
        self.key = nn.Linear(hidden_dim*2, hidden_dim)
        self.value = nn.Linear(hidden_dim*2, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).item()
        
    def forward(self, audio_features, face_features):
        # audio_features shape: (batch_size, seq_len, hidden_dim*2)
        # face_features shape: (batch_size, seq_len, hidden_dim*2)
        
        Q = self.query(face_features)
        K = self.key(audio_features)
        V = self.value(audio_features)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        
        # Concatenate with face features for residual connection
        return torch.cat((x, face_features), dim=2)

class LipSyncModel(nn.Module):
    def __init__(self, input_dim=13, face_dim=40, hidden_dim=64, output_dim=40):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.audio_encoder = AudioEncoder(input_dim, hidden_dim)
        self.face_encoder = FaceEncoder(face_dim, hidden_dim)
        self.attention = AttentionLayer(hidden_dim)
        self.face_decoder = FaceDecoder(hidden_dim*4, hidden_dim, output_dim)
        
    def forward(self, x, face_input=None):
        # x shape: (batch_size, seq_len, input_dim) - MFCC features
        
        # Encode audio
        audio_features = self.audio_encoder(x)
        
        # If face input is provided (for training with teacher forcing)
        if face_input is not None:
            face_features = self.face_encoder(face_input)
        else:
            # or previous predictions in a more advanced setup
            batch_size, seq_len, _ = x.shape
            face_features = torch.zeros(batch_size, seq_len, self.hidden_dim*2, device=x.device)
        # Apply attention mechanism
        combined_features = self.attention(audio_features, face_features)
        
        # Decode to get facial landmarks
        landmarks = self.face_decoder(combined_features)
        
        return landmarks  # (batch, seq_len, output_dim)
    
    def inference(self, audio_input):
        """
        Simplified inference method that only requires audio input
        """
        return self.forward(audio_input)
