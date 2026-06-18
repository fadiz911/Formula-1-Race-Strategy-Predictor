import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DriverTransformer(nn.Module):
    def __init__(self, num_teams, num_tracks, embedding_dim=16, d_model=32, nhead=2, num_layers=1):
        super(DriverTransformer, self).__init__()
        
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)
        
        # Concatenate features and embeddings: 12 features + 16 team + 16 track = 44 input size
        self.input_proj = nn.Linear(12 + (embedding_dim * 2), d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Current features: 8 features + 16 team + 16 track = 40 input size
        self.current_feature_dim = 8 + (embedding_dim * 2)
        
        # Combined prediction head
        self.fc1 = nn.Linear(d_model + self.current_feature_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, history_seq, current_features, history_teams, history_tracks, current_team, current_track):
        # 1. Process History Sequence
        seq_team_emb = self.team_embedding(history_teams)
        seq_track_emb = self.track_embedding(history_tracks)
        
        # Concatenate features and embeddings
        seq_input = torch.cat([history_seq, seq_team_emb, seq_track_emb], dim=-1)
        seq_proj = self.input_proj(seq_input)
        
        # Transpose to [SeqLen, Batch, d_model] and pass to Transformer
        x = seq_proj.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        context_vector = x[-1] # [Batch, d_model]
        
        # 2. Process Current GP
        curr_team_emb = self.team_embedding(current_team)
        curr_track_emb = self.track_embedding(current_track)
        
        # 3. Concatenate context + current features/embeddings
        combined = torch.cat([context_vector, current_features, curr_team_emb, curr_track_emb], dim=1)
        
        # 4. Prediction Head
        fused = self.relu(self.fc1(combined))
        fused = self.dropout(fused)
        out = self.fc2(fused)
        return out
