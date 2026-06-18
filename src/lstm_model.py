
import torch
import torch.nn as nn

class DriverLSTM(nn.Module):
    def __init__(self, num_teams, num_tracks, embedding_dim=8, hidden_dim=32, num_layers=1):
        super(DriverLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)
        
        # Sequence Features: [FinishPos, GridPos, Points, SpeedST, StintCount, QualiDelta, PracticePace, IsWet, DriverConsistency, TrackPerformance, TeamAvgPoints, ReliabilityRisk]
        self.seq_input_dim = 12 + (embedding_dim * 2)
        
        # LSTM Layer
        self.lstm = nn.LSTM(self.seq_input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Current Race Features: [GridPos, QualiDelta, PracticePace, IsWet, DriverConsistency, TrackPerformance, TeamAvgPoints, ReliabilityRisk] + Embeddings
        self.current_feature_dim = 8 + (embedding_dim * 2)
        
        # Combined Head
        self.fc1 = nn.Linear(hidden_dim + self.current_feature_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, history_seq, current_features, history_teams, history_tracks, current_team, current_track):
        # --- 1. Process History Sequence ---
        # history_seq: [Batch, SeqLen, 3] (Finish, Grid, Points)
        # history_teams: [Batch, SeqLen]
        # history_tracks: [Batch, SeqLen]
        
        batch_size = history_seq.size(0)
        
        # Embeddings for sequence
        seq_team_emb = self.team_embedding(history_teams) # [Batch, SeqLen, Emb]
        seq_track_emb = self.track_embedding(history_tracks) # [Batch, SeqLen, Emb]
        
        # Concat embeddings with numerical features
        lstm_input = torch.cat([history_seq, seq_team_emb, seq_track_emb], dim=2) # [Batch, SeqLen, 19]
        
        # LSTM Pass
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Take the last hidden state (context vector)
        context_vector = h_n[-1] # [Batch, Hidden]
        
        # --- 2. Process Current Race ---
        # current_features: [Batch, 1] (GridPos)
        curr_team_emb = self.team_embedding(current_team) # [Batch, Emb]
        curr_track_emb = self.track_embedding(current_track) # [Batch, Emb]
        
        # Concat context + current
        combined = torch.cat([context_vector, current_features, curr_team_emb, curr_track_emb], dim=1)
        
        # --- 3. Prediction Head ---
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
