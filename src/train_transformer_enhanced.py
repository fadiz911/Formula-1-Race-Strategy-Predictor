import fastf1
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle
import logging
from src.transformer_model import DriverTransformer
from src.train_lstm_enhanced import get_enhanced_race_data

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = 'models_transformer'
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

def train_transformer():
    print("🚀 Starting Upgraded Transformer Training...")
    
    # 1. Load Data (2022-2026)
    df = get_enhanced_race_data(years=[2022, 2023, 2024, 2025, 2026])
    if df.empty: return
    
    df = df.sort_values(['Year', 'Round'])
    
    # 2. Encoders & Scalers
    le_team = LabelEncoder()
    df['TeamId'] = le_team.fit_transform(df['Team'].astype(str))
    
    le_track = LabelEncoder()
    df['TrackId'] = le_track.fit_transform(df['Track'].astype(str))
    
    # Data Cleaning
    df['FinishPos'] = df['FinishPos'].fillna(20.0)
    df['GridPos'] = df['GridPos'].fillna(20.0)
    df['Points'] = df['Points'].fillna(0.0)
    df['SpeedST'] = df['SpeedST'].fillna(300.0)
    df['StintCount'] = df['StintCount'].fillna(1.0)
    df['QualiDelta'] = df['QualiDelta'].replace([np.inf, -np.inf], 2.0).fillna(2.0)
    df['PracticePace'] = df['PracticePace'].replace([np.inf, -np.inf], 0.5).fillna(0.5)
    df['IsWet'] = df['IsWet'].fillna(0)
    df['DriverConsistency'] = df['DriverConsistency'].replace([np.inf, -np.inf], 5.0).fillna(5.0)
    df['TrackPerformance'] = df['TrackPerformance'].replace([np.inf, -np.inf], 10.5).fillna(10.5)
    
    # Store a copy of unscaled target to prevent leakage & matching predictor unscale expectation
    df['FinishPos_Unscaled'] = df['FinishPos']
    
    # Scale Numerical Features (12 features)
    scaler = StandardScaler()
    feature_cols = ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 
                    'QualiDelta', 'PracticePace', 'IsWet', 'DriverConsistency', 'TrackPerformance', 'TeamAvgPoints', 'ReliabilityRisk']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save Artifacts
    with open(os.path.join(MODEL_DIR, 'artifacts.pkl'), 'wb') as f:
        pickle.dump({
            'encoders': {'team': le_team, 'track': le_track},
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)
        
    # 3. Create Sequences
    sequences = []
    targets = []
    
    hist_team_seq = []
    hist_track_seq = []
    
    curr_feat_seq = []
    curr_team_seq = []
    curr_track_seq = []
    
    SEQ_LEN = 5
    
    for driver in df['Driver'].unique():
        d_df = df[df['Driver'] == driver]
        if len(d_df) < SEQ_LEN + 1: continue
        
        feat_vals = d_df[feature_cols].values
        team_vals = d_df['TeamId'].values
        track_vals = d_df['TrackId'].values
        unscaled_finishes = d_df['FinishPos_Unscaled'].values
        
        for i in range(len(d_df) - SEQ_LEN):
            seq_x = feat_vals[i : i+SEQ_LEN]
            seq_team = team_vals[i : i+SEQ_LEN]
            seq_track = track_vals[i : i+SEQ_LEN]
            
            target_idx = i + SEQ_LEN
            target_y = feat_vals[target_idx][0] # FinishPos (Scaled)
            
            curr_grid = feat_vals[target_idx][1]
            curr_quali = feat_vals[target_idx][5]
            curr_pace = feat_vals[target_idx][6]
            curr_wet = feat_vals[target_idx][7]
            curr_consistency = feat_vals[target_idx][8]
            curr_track_perf = feat_vals[target_idx][9]
            curr_team_avg = feat_vals[target_idx][10]
            curr_reliability = feat_vals[target_idx][11]
            
            curr_feat = [curr_grid, curr_quali, curr_pace, curr_wet, curr_consistency, curr_track_perf, curr_team_avg, curr_reliability]
            
            curr_team = team_vals[target_idx]
            curr_track = track_vals[target_idx]
            
            sequences.append(seq_x)
            hist_team_seq.append(seq_team)
            hist_track_seq.append(seq_track)
            
            curr_feat_seq.append(curr_feat)
            curr_team_seq.append(curr_team)
            curr_track_seq.append(curr_track)
            
            targets.append(target_y)
            
    # Convert to Tensors
    X_seq = torch.tensor(np.array(sequences), dtype=torch.float32)
    X_team_hist = torch.tensor(np.array(hist_team_seq), dtype=torch.long)
    X_track_hist = torch.tensor(np.array(hist_track_seq), dtype=torch.long)
    
    X_curr_feat = torch.tensor(np.array(curr_feat_seq), dtype=torch.float32)
    X_curr_team = torch.tensor(np.array(curr_team_seq), dtype=torch.long)
    X_curr_track = torch.tensor(np.array(curr_track_seq), dtype=torch.long)
    
    y = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)
    
    # 4. Train Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DriverTransformer(
        num_teams=len(le_team.classes_),
        num_tracks=len(le_track.classes_),
        embedding_dim=16,
        d_model=32,
        nhead=2,
        num_layers=2
    ).to(device)
    
    # Move tensors to device for full batch training
    X_seq = X_seq.to(device)
    X_curr_feat = X_curr_feat.to(device)
    X_team_hist = X_team_hist.to(device)
    X_track_hist = X_track_hist.to(device)
    X_curr_team = X_curr_team.to(device)
    X_curr_track = X_curr_track.to(device)
    y = y.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss(delta=2.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    epochs = 300
    print(f"Training on {len(y)} sequences for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(X_seq, X_curr_feat, X_team_hist, X_track_hist, X_curr_team, X_curr_track)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss.item())
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")
            
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'transformer_model.pth'))
    print("✅ Upgraded Transformer Model Saved successfully!")

if __name__ == "__main__":
    train_transformer()
