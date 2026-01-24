import fastf1
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle
import logging
from src.lstm_model import DriverLSTM

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = 'models_dl'
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists('cache'): os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

def get_enhanced_race_data(years=[2022, 2023, 2024, 2025]):
    """
    Fetches race data with V3 Features:
    - QualiDelta (Gap to Pole)
    - PracticePace (FP2 Long Run)
    - IsWet (Rain Flag)
    - SpeedST (Top Speed)
    """
    all_data = []
    
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] == 'conventional']
            
            # OPTIMIZATION: For 2022, slice to last 5 rounds for History Context
            if year == 2022:
                 races = races.tail(5)
                 logger.info(f"Optimization: Loading ONLY last 5 rounds of {year} for history context.")
            
            for _, race in races.iterrows():
                if race['EventName'] == 'TOSA': continue # Skip testing
                
                # Check for future races (prevent fetching non-existent data)
                # Removed hardcoded limits for 2024/2025 as they have occurred now.

                try:
                    logger.info(f"Processing {year} {race['EventName']}...")
                    
                    # 1. LOAD SESSIONS
                    # RACE
                    session_r = fastf1.get_session(year, race['RoundNumber'], 'R')
                    session_r.load(laps=True, telemetry=False, weather=True, messages=False)
                    
                    # QUALIFYING (For Pace Delta)
                    try:
                        session_q = fastf1.get_session(year, race['RoundNumber'], 'Q')
                        session_q.load(laps=True, telemetry=False, weather=False, messages=False)
                        pole_lap = session_q.laps.pick_fastest()['LapTime'].total_seconds()
                    except:
                        session_q = None
                        pole_lap = None

                    # PRACTICE (FP2 for Race Pace)
                    try:
                        session_fp = fastf1.get_session(year, race['RoundNumber'], 'FP2')
                        session_fp.load(laps=True, telemetry=False, weather=False, messages=False)
                    except:
                        session_fp = None
                    
                    if session_r.laps.empty: continue
                    
                    # Weather Feature
                    is_wet = 1 if session_r.weather_data['Rainfall'].max() > 0 else 0
                    
                    # Field Averages for Practice
                    avg_fp_pace = 90.0
                    if session_fp and not session_fp.laps.empty:
                        # Clean outliers (> 107% rule roughly)
                        valid_laps = session_fp.laps.pick_quicklaps()
                        avg_fp_pace = valid_laps['LapTime'].dt.total_seconds().mean()

                    # Extract Features per Driver
                    for driver in session_r.results['Abbreviation'].unique():
                        try:
                            d_laps_r = session_r.laps[session_r.laps['Driver'] == driver]
                            if d_laps_r.empty: continue
                            
                            # Result Info
                            res = session_r.results[session_r.results['Abbreviation'] == driver].iloc[0]
                            # We keep DNFs but maybe mark them? For now, standard training.
                            
                            # FEATURE 1: Quali Delta (Gap to Pole)
                            quali_delta = 0.0
                            if session_q and pole_lap:
                                try:
                                    d_q_lap = session_q.laps.pick_drivers(driver).pick_fastest()['LapTime'].total_seconds()
                                    quali_delta = d_q_lap - pole_lap
                                except:
                                    # No time set or crash
                                    quali_delta = 2.0 # Penalty default
                            
                            # FEATURE 2: Practice Pace (vs Field Avg)
                            practice_diff = 0.0
                            if session_fp:
                                try:
                                    # Calculate avg run (excluding in/out laps)
                                    d_fp = session_fp.laps.pick_drivers(driver).pick_quicklaps()
                                    if not d_fp.empty:
                                        d_avg = d_fp['LapTime'].dt.total_seconds().mean()
                                        practice_diff = d_avg - avg_fp_pace
                                    else:
                                        practice_diff = 0.5 # Slower default
                                except:
                                    practice_diff = 0.5
                            
                            # Feature 3: Light Telemetry
                            speed_st = d_laps_r['SpeedST'].max()
                            if pd.isna(speed_st): speed_st = d_laps_r['SpeedI1'].max()
                            if pd.isna(speed_st): speed_st = 300.0 # Fallback
                            
                            stint_count = d_laps_r['Stint'].nunique()
                            
                            all_data.append({
                                'Year': year,
                                'Round': race['RoundNumber'],
                                'Track': race['EventName'],
                                'Driver': driver,
                                'Team': res['TeamName'],
                                'GridPos': res['GridPosition'],
                                'FinishPos': res['Position'],
                                'Points': res['Points'],
                                'SpeedST': speed_st,
                                'StintCount': stint_count,
                                'QualiDelta': quali_delta,   # NEW
                                'PracticePace': practice_diff, # NEW
                                'IsWet': is_wet              # NEW
                            })
                        except Exception as e:
                            pass
                            
                except Exception as e:
                    logger.error(f"Failed {race['EventName']}: {e}")
        except Exception as e:
            logger.error(f"Failed year {year}: {e}")
            
    # Convert to DataFrame FIRST for feature engineering
    df = pd.DataFrame(all_data)
    
    # ===== PHASE 2 FEATURE ENGINEERING =====
    print("📊 Phase 2: Computing advanced features...")
    
    # Feature 9: Driver Consistency Score (Rolling Std Dev of last 5 finishes)
    # Lower = more consistent (Verstappen), Higher = volatile (Stroll)
    df = df.sort_values(['Year', 'Round'])
    df['DriverConsistency'] = df.groupby('Driver')['FinishPos'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(5.0)
    )
    
    # Feature 10: Track-Specific Driver Performance (Avg finish at this track)
    # Hamilton at Silverstone vs Hamilton at Monaco
    track_performance = df.groupby(['Driver', 'Track'])['FinishPos'].transform('mean')
    df['TrackPerformance'] = track_performance.fillna(10.5)  # Midfield default
    
    print(f"✅ Added DriverConsistency and TrackPerformance features")
    
    return df

def train_lstm():
    print("🚀 Starting Enhanced LSTM Training...")
    
    # 1. Load Data
    df = get_enhanced_race_data(years=[2022, 2023, 2024, 2025])
    if df.empty: return
    
    df = df.sort_values(['Year', 'Round'])
    
    # 2. Encoders & Scalers
    le_team = LabelEncoder()
    df['TeamId'] = le_team.fit_transform(df['Team'].astype(str))
    
    le_track = LabelEncoder()
    df['TrackId'] = le_track.fit_transform(df['Track'].astype(str))
    
    # 2.5 Data Cleaning (CRITICAL for V3)
    # Fill missing values to prevent Loss=NaN
    df['FinishPos'] = df['FinishPos'].fillna(20.0) # DNF -> P20
    df['GridPos'] = df['GridPos'].fillna(20.0)
    df['Points'] = df['Points'].fillna(0.0)
    df['SpeedST'] = df['SpeedST'].fillna(300.0)
    df['StintCount'] = df['StintCount'].fillna(1.0)
    
    # V3 Features
    df['QualiDelta'] = df['QualiDelta'].replace([np.inf, -np.inf], 2.0).fillna(2.0)
    df['PracticePace'] = df['PracticePace'].replace([np.inf, -np.inf], 0.5).fillna(0.5)
    df['IsWet'] = df['IsWet'].fillna(0)
    
    # Phase 2 Features (V4)
    df['DriverConsistency'] = df['DriverConsistency'].replace([np.inf, -np.inf], 5.0).fillna(5.0)
    df['TrackPerformance'] = df['TrackPerformance'].replace([np.inf, -np.inf], 10.5).fillna(10.5)
    
    # Scale Numerical Features
    # Recalibrate scaler for Phase 2 (10 features)
    # Input Features: [FinishPos, GridPos, Points, SpeedST, StintCount, QualiDelta, PracticePace, IsWet, DriverConsistency, TrackPerformance]
    scaler = StandardScaler()
    feature_cols = ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 
                    'QualiDelta', 'PracticePace', 'IsWet', 'DriverConsistency', 'TrackPerformance']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save Artifacts
    with open(os.path.join(MODEL_DIR, 'lstm_artifacts_v3.pkl'), 'wb') as f:
        pickle.dump({
            'encoders': {'team': le_team, 'track': le_track},
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)
        
    # 3. Create Sequences
    # Group by Driver, then sliding window of 5
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
        
        # Values - NOW 10 features
        feat_vals = d_df[feature_cols].values
        team_vals = d_df['TeamId'].values
        track_vals = d_df['TrackId'].values
        
        for i in range(len(d_df) - SEQ_LEN):
            # Input: T_0 to T_4 (All 10 features)
            seq_x = feat_vals[i : i+SEQ_LEN]
            seq_team = team_vals[i : i+SEQ_LEN]
            seq_track = track_vals[i : i+SEQ_LEN]
            
            # Target: T_5 (Finish Position)
            target_idx = i + SEQ_LEN
            target_y = feat_vals[target_idx][0] # FinishPos (Scaled)
            
            # Current Race Inputs (Pre-race knowns) - Phase 2: 6 features
            # Feature Indices: [FinishPos=0, GridPos=1, Points=2, SpeedST=3, StintCount=4, 
            #                   QualiDelta=5, PracticePace=6, IsWet=7, DriverConsistency=8, TrackPerformance=9]
            curr_grid = feat_vals[target_idx][1] # GridPos
            curr_quali = feat_vals[target_idx][5] # QualiDelta
            curr_pace = feat_vals[target_idx][6] # PracticePace
            curr_wet = feat_vals[target_idx][7] # IsWet
            curr_consistency = feat_vals[target_idx][8] # NEW: DriverConsistency
            curr_track_perf = feat_vals[target_idx][9] # NEW: TrackPerformance
            
            curr_team = team_vals[target_idx]
            curr_track = track_vals[target_idx]
            
            sequences.append(seq_x)
            hist_team_seq.append(seq_team)
            hist_track_seq.append(seq_track)
            
            curr_feat_seq.append([curr_grid, curr_quali, curr_pace, curr_wet, curr_consistency, curr_track_perf])
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
    # DriverLSTM updated automatically for 8 features in logic, 
    # but we should ensure hidden_dim is enough for complexity.
    model = DriverLSTM(
        num_teams=len(le_team.classes_),
        num_tracks=len(le_track.classes_),
        hidden_dim=64,
        num_layers=2
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Phase 1 Optimization: Huber Loss (robust to DNF outliers)
    criterion = nn.HuberLoss(delta=2.0)  # Was MSELoss
    # Phase 1 Optimization: LR Scheduler for fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    print(f"Training on {len(y)} sequences...")
    # Phase 1 Optimization: Increased epochs 150 -> 300
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        
        out = model(X_seq, X_curr_feat, X_team_hist, X_track_hist, X_curr_team, X_curr_track)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        # Update LR Scheduler
        scheduler.step(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'lstm_model_v3.pth'))
    print("✅ Enhanced LSTM Model V3 Saved!")

if __name__ == "__main__":
    train_lstm()
