
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class F1SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return tuple: (hist_num, hist_team, hist_track, curr_num, curr_team, curr_track, target)
        return self.sequences[idx]

def create_sequences(df, seq_length=5):
    """
    Transforms flat race data into driver-specific sequences.
    
    df columns expected: 
    ['Driver', 'Year', 'Round', 'GridPos', 'FinishPos', 'Points', 'TeamId', 'TrackId']
    """
    sequences = []
    
    # Sort critical for time dependence
    df = df.sort_values(['Year', 'Round'])
    
    # Process per driver
    for driver in df['Driver'].unique():
        d_data = df[df['Driver'] == driver].copy()
        d_data = d_data.sort_values(['Year', 'Round'])
        
        # Convert to list of dicts for easy indexing
        records = d_data.to_dict('records')
        
        # We need at least seq_length + 1 races (history + target)
        if len(records) < seq_length + 1:
            continue
            
        for i in range(len(records) - seq_length):
            # History window: i to i+seq_length-1
            history = records[i : i+seq_length]
            
            # Target race: i+seq_length
            target_race = records[i+seq_length]
            
            # --- Build Tensors ---
            
            # History Numerical: [Finish, Grid, Points]
            hist_num = np.array([[r['FinishPos'], r['GridPos'], r['Points']] for r in history], dtype=np.float32)
            
            # History Categorical
            hist_team = np.array([r['TeamId'] for r in history], dtype=np.int64)
            hist_track = np.array([r['TrackId'] for r in history], dtype=np.int64)
            
            # Current Numerical: [GridPos] (Finish and Points are unknown for today!)
            curr_num = np.array([target_race['GridPos']], dtype=np.float32)
            
            # Current Categorical
            curr_team = np.array(target_race['TeamId'], dtype=np.int64)
            curr_track = np.array(target_race['TrackId'], dtype=np.int64)
            
            # Target
            target = np.array([target_race['FinishPos']], dtype=np.float32)
            
            # Store tuple
            # Convert to torch tensors now or in collation? Let's do numpy here, Torch in Dataset
            sequences.append((
                torch.tensor(hist_num),
                torch.tensor(hist_team), 
                torch.tensor(hist_track),
                torch.tensor(curr_num),
                torch.tensor(curr_team),
                torch.tensor(curr_track),
                torch.tensor(target)
            ))
            
    return sequences
