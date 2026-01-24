
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import sys

# Import our custom modules
from src.dl_training import get_race_data
from src.sequence_loader import create_sequences, F1SequenceDataset
from src.lstm_model import DriverLSTM

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        # Unpack
        (h_seq, h_team, h_track, c_feat, c_team, c_track, target) = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        preds = model(h_seq, c_feat, h_team, h_track, c_team, c_track)
        loss = criterion(preds, target.view(-1, 1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds_all = []
    targets_all = []
    
    with torch.no_grad():
        for batch in loader:
             (h_seq, h_team, h_track, c_feat, c_team, c_track, target) = [b.to(device) for b in batch]
             preds = model(h_seq, c_feat, h_team, h_track, c_team, c_track)
             
             preds_all.extend(preds.cpu().numpy().flatten())
             targets_all.extend(target.cpu().numpy().flatten())
             
    return preds_all, targets_all

def run_walk_forward_validation():
    print("🚀 Starting LSTM Walk-Forward Validation (2024 Replay)...")
    
    # 1. Load Data
    df = get_race_data(years=[2023, 2024])
    if df.empty:
        print("No data found.")
        return

    # 2. Preprocessing
    # Sort
    df = df.sort_values(['Year', 'Round', 'FinishPos'])
    
    # Encoders
    le_team = LabelEncoder()
    df['TeamId'] = le_team.fit_transform(df['Team'])
    num_teams = len(le_team.classes_)
    
    le_track = LabelEncoder()
    df['TrackId'] = le_track.fit_transform(df['Track'])
    num_tracks = len(le_track.classes_)
    
    # Scaling (Numerical features only)
    # Fit scaler on 2023 only to prevent leakage!
    scaler = StandardScaler()
    df_2023 = df[df['Year'] == 2023]
    scaler.fit(df_2023[['FinishPos', 'GridPos', 'Points']])
    
    # Transform whole DF
    # Note: We scale columns temporarily for sequence generation
    # But we must be careful: Sequences store raw tensor values? 
    # Better to scale the columns in the DF before sequence gen
    df_scaled = df.copy()
    df_scaled[['FinishPos', 'GridPos', 'Points']] = scaler.transform(df[['FinishPos', 'GridPos', 'Points']])
    
    # 3. Create Sequences
    # This creates a big list of (History, Target) tuples
    all_sequences = create_sequences(df_scaled, seq_length=5)
    print(f"Total Sequences Generated: {len(all_sequences)}")
    
    # 4. Split by Time
    # We identify which sequences belong to 2023 (Initial Train) and 2024 (Test Loop)
    # The 'create_sequences' function doesn't return metadata, so we need a mapping strategy.
    # Actually, simplistic approach: 
    #   We iterate through rounds of 2024.
    #   For each round, we identify the sequences that correspond to that round.
    #   How? The 'Target' in the sequence is the finish pos of that race.
    #   Implementing a robust filter is tricky without metadata in the sequence object.
    
    # BETTER APPROACH: 
    # Keep the raw data around.
    # For the simulation loop:
    #   Train Set = All sequences where Target Race < 2024 Round 1.
    #   Loop 2024 Rounds:
    #       Current Test Set = Sequences where Target Race == This Round.
    #       Train Model on Train Set.
    #       Predict Test Set.
    #       Add Test Set to Train Set.
    
    # To do this, 'create_sequences' needs to optionally return metadata or we keep index alignment.
    # Let's Modify the loader logic slightly inside here or re-parse.
    # Quick fix: Re-generate sequences every loop? No, too slow.
    # Optimization: Store (Year, Round) in the sequence tuple for filtering?
    # Yes, let's modify create_sequences in memory here or monkey patch?
    # No, let's just use the fact that we can filter the source DF.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    model = DriverLSTM(num_teams, num_tracks, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Initial Train Set: 2023 sequences
    # We can just filter the DF for 2023, generate sequences.
    df_train_pool = df_scaled[df_scaled['Year'] == 2023].copy()
    
    # Train Loop
    print("\nPhase 1: Pre-training on 2023 Season...")
    train_seqs = create_sequences(df_train_pool, seq_length=5)
    train_loader = DataLoader(F1SequenceDataset(train_seqs), batch_size=32, shuffle=True)
    
    for epoch in range(50):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss {loss:.4f}")
            
    # Walk-Forward Loop (2024)
    rounds_2024 = sorted(df[df['Year'] == 2024]['Round'].unique())
    
    spearman_corrs = []
    
    print("\nPhase 2: Walk-Forward Validation (2024)...")
    for r in rounds_2024:
        # 1. Define Test Data (Current Round)
        # We need sequences where the TARGET is this round.
        # This implies history is valid (previous races).
        
        # We take the whole history up to this round
        df_current_context = df_scaled[
            ((df_scaled['Year'] == 2023)) | 
            ((df_scaled['Year'] == 2024) & (df_scaled['Round'] <= r))
        ]
        
        # Generate ALL sequences from this context
        # Then filter for only those ending in the current round
        # This is inefficient but safe.
        # Filtering rule: The LAST item in the sequence logic (target) must be this round.
        # We can simulate this by generating sequences for JUST this round's drivers
        # BUT we need their history.
        
        # Optimized: Generate for all, filter by "Target is Round R"? 
        # Since our sequence loader is opaque, let's trust the "Add to Train Pool" method.
        
        # Add THIS round's rows to the pool, but BEFORE we train on them, we predict them?
        # No, we can't add them to history yet if we want to predict them.
        # We need their history.
        
        # Get drivers in this round
        round_drivers = df_scaled[(df_scaled['Year'] == 2024) & (df_scaled['Round'] == r)]['Driver'].unique()
        
        batch_seqs = []
        
        for d in round_drivers:
            # get driver history
            d_hist = df_scaled[
                (df_scaled['Driver'] == d) & 
                (
                    (df_scaled['Year'] == 2023) | 
                    ((df_scaled['Year'] == 2024) & (df_scaled['Round'] < r))
                )
            ].sort_values(['Year', 'Round'])
            
            if len(d_hist) < 5: continue # Not enough history
            
            # Get last 5
            hist_recs = d_hist.tail(5).to_dict('records')
            
            # Get current target
            target_rec = df_scaled[
                (df_scaled['Driver'] == d) & 
                (df_scaled['Year'] == 2024) & 
                (df_scaled['Round'] == r)
            ].iloc[0]
            
            # Build Tuple (Manual to match loader logic)
            hist_num = np.array([[x['FinishPos'], x['GridPos'], x['Points']] for x in hist_recs], dtype=np.float32)
            hist_team = np.array([x['TeamId'] for x in hist_recs], dtype=np.int64)
            hist_track = np.array([x['TrackId'] for x in hist_recs], dtype=np.int64)
            
            target = np.array([target_rec['FinishPos']], dtype=np.float32)
            
            # Current Feats
            curr_num = np.array([target_rec['GridPos']], dtype=np.float32)
            curr_team = np.array(target_rec['TeamId'], dtype=np.int64)
            curr_track = np.array(target_rec['TrackId'], dtype=np.int64)
            
            batch_seqs.append((
                torch.tensor(hist_num), torch.tensor(hist_team), torch.tensor(hist_track),
                torch.tensor(curr_num), torch.tensor(curr_team), torch.tensor(curr_track),
                torch.tensor(target)
            ))
            
        if not batch_seqs: continue
        
        # Predict
        test_loader = DataLoader(F1SequenceDataset(batch_seqs), batch_size=len(batch_seqs))
        preds, actuals = evaluate(model, test_loader, device)
        
        # Calc Correlation
        if len(preds) > 5:
            df_res = pd.DataFrame({'Pred': preds, 'Act': actuals})
            # Inverse transform to get real positions?
            # Actually spearman is rank-invariant, so scaling doesn't matter much.
            corr = df_res['Pred'].corr(df_res['Act'], method='spearman')
            spearman_corrs.append(corr)
            print(f"Round {r}: Spearman {corr:.2f} ({len(preds)} drivers)")
        
        # Online Training: Add this round to the knowledge base
        # effectively, we perform a few backprop steps on this new batch
        # to "Learn" the result of this race before the next one.
        
        # We can also retrain on the WHOLE history + this new batch.
        # Let's do a quick fine-tune on the new batch (Online Learning) + general replay
        
        # Fine tune heavily on recent data
        ft_loader = DataLoader(F1SequenceDataset(batch_seqs), batch_size=16, shuffle=True)
        for _ in range(5):
            train_epoch(model, ft_loader, criterion, optimizer, device)
            
    avg_corr = np.mean(spearman_corrs)
    print(f"\n✅ Walk-Forward Complete.")
    print(f"Average 2024 Correlation: {avg_corr:.4f}")
    
    # SAVE MODEL FOR STREAMLIT
    import os
    import pickle
    
    MODEL_DIR = 'models_dl'
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    # Save Weights
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'lstm_model.pth'))
    
    # Save Artifacts (Encoders, Scaler)
    with open(os.path.join(MODEL_DIR, 'lstm_artifacts.pkl'), 'wb') as f:
        pickle.dump({
            'encoders': {'team': le_team, 'track': le_track},
            'scaler': scaler
        }, f)
        
    print("✅ LSTM Model & Artifacts Saved for Production!")

if __name__ == "__main__":
    run_walk_forward_validation()
