
import torch
import numpy as np
import pandas as pd
import os
import pickle
import logging
from src.lstm_model import DriverLSTM
from src.train_lstm_enhanced import get_enhanced_race_data  # Use new data loader

logger = logging.getLogger(__name__)

class LSTMPredictor:
    def __init__(self, model_dir='models_dl'):
        self.model_path = os.path.join(model_dir, 'lstm_model_v3.pth') # V4 Model with Phase 2 features
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.encoders = None
        self.scaler = None
        self.history_df = pd.DataFrame()
        self.loaded = False
        
        # Load artifacts
        self.encoders_path = os.path.join(model_dir, 'lstm_artifacts_v3.pkl') 
        self._load_artifacts()
        
    def _load_artifacts(self):
        if os.path.exists(self.encoders_path):
             with open(self.encoders_path, 'rb') as f:
                data = pickle.load(f)
                self.encoders = data['encoders']
                self.scaler = data['scaler']
                # Load Model
                self.model = DriverLSTM(
                    num_teams=len(self.encoders['team'].classes_), 
                    num_tracks=len(self.encoders['track'].classes_),
                    hidden_dim=64,
                    num_layers=2 
                ).to(self.device)
                
                if os.path.exists(self.model_path):
                    try:
                        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                        self.model.eval()
                        self.loaded = True
                    except RuntimeError as e:
                        logger.error(f"Model dimension mismatch (Architecture changed?): {e}")
                        print(f"⚠️ Warning: Could not load LSTM weights due to mismatch. Using 'Physics Only' mode.")
                        self.loaded = False
                    
                    # Pre-load history for sequence generation
                    try: 
                        cache_path = os.path.join('cache', 'history_clean.pkl')
                        if os.path.exists(cache_path):
                            with open(cache_path, 'rb') as f:
                                self.history_df = pickle.load(f)
                        else:
                            # Fetch new data with 12 features (2022-2026 for full context)
                            self.history_df = get_enhanced_race_data(years=[2022, 2023, 2024, 2025, 2026])
                            if not self.history_df.empty:
                                os.makedirs('cache', exist_ok=True)
                                with open(cache_path, 'wb') as f:
                                    pickle.dump(self.history_df, f)
                    except:
                        self.history_df = pd.DataFrame() 
                        
                    # Pre-scale
                    if not self.history_df.empty:
                        self.history_df = self.history_df.sort_values(['Year', 'Round'])
                        
                        # Normalize Team Names
                        def normalize_team(t):
                            t = str(t)
                            if 'Alfa Romeo' in t: return 'Kick Sauber'
                            if 'AlphaTauri' in t: return 'RB'
                            if 'Aston Martin' in t: return 'Aston Martin'
                            return t
                            
                        self.history_df['Team'] = self.history_df['Team'].apply(normalize_team)
                        
                        # Filter teams/tracks
                        known_teams = set(self.encoders['team'].classes_)
                        self.history_df = self.history_df[self.history_df['Team'].isin(known_teams)]
                        
                        known_tracks = set(self.encoders['track'].classes_)
                        self.history_df = self.history_df[self.history_df['Track'].isin(known_tracks)]
                        
                        self.history_df['TeamId'] = self.encoders['team'].transform(self.history_df['Team'].astype(str))
                        self.history_df['TrackId'] = self.encoders['track'].transform(self.history_df['Track'].astype(str))
                        
                        # Apply scaling to 12 columns
                        scaler_cols = ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 
                                       'QualiDelta', 'PracticePace', 'IsWet', 'DriverConsistency', 'TrackPerformance', 'TeamAvgPoints', 'ReliabilityRisk']
                        self.history_df[scaler_cols] = self.scaler.transform(self.history_df[scaler_cols])
                        
                else:
                    logger.warning("LSTM weights not found.")
        else:
            logger.warning("LSTM artifacts not found.")

    @property
    def raw_history_df(self):
        if self.history_df.empty:
            return self.history_df
        df = self.history_df.copy()
        scaler_cols = ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 
                       'QualiDelta', 'PracticePace', 'IsWet', 'DriverConsistency', 'TrackPerformance', 'TeamAvgPoints', 'ReliabilityRisk']
        df[scaler_cols] = self.scaler.inverse_transform(df[scaler_cols])
        return df

    def predict_position(self, driver, year, round_num, current_grid, current_team_name, current_track_name, context={}):
        if not self.loaded or self.history_df.empty: return None
        
        try:
            # 1. Get History
            d_hist = self.history_df[
                (self.history_df['Driver'] == driver) & 
                (
                    (self.history_df['Year'] < year) | 
                    ((self.history_df['Year'] == year) & (self.history_df['Round'] < round_num))
                )
            ].sort_values(['Year', 'Round'])
            
            if len(d_hist) < 5: return None 
            
            hist_recs = d_hist.tail(5).to_dict('records')
            
            # Prepare Tensor (12 Features)
            # Data is already scaled
            hist_vals = np.array([[
                x['FinishPos'], x['GridPos'], x['Points'], x['SpeedST'], x['StintCount'],
                x['QualiDelta'], x['PracticePace'], x['IsWet'], x['DriverConsistency'], x['TrackPerformance'], x['TeamAvgPoints'], x['ReliabilityRisk']
            ] for x in hist_recs])
            
            # Fix UserWarning: Convert to np.array first
            hist_num = torch.tensor(hist_vals[np.newaxis, :, :], dtype=torch.float32).to(self.device) 
            hist_team = torch.tensor(np.array([[x['TeamId'] for x in hist_recs]]), dtype=torch.long).to(self.device)
            hist_track = torch.tensor(np.array([[x['TrackId'] for x in hist_recs]]), dtype=torch.long).to(self.device)
            
            # 2. Current Features
            try: t_id = self.encoders['team'].transform([str(current_team_name)])[0]
            except: t_id = 0
            try: tr_id = self.encoders['track'].transform([str(current_track_name)])[0]
            except: tr_id = 0
            
            # Extract Context or Defaults
            q_delta = context.get('QualiDelta', 0.0)
            p_pace = context.get('PracticePace', 0.0)
            is_wet = context.get('IsWet', 0.0)
            driver_cons = context.get('DriverConsistency', 5.0)
            track_perf = context.get('TrackPerformance', 10.5)
            team_avg = context.get('TeamAvgPoints', 5.0)
            reliability = context.get('ReliabilityRisk', 0.1)
            
            # Scale Current Inputs
            # Create dummy row matching scaler shape (12 cols)
            scaler_cols = ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 'QualiDelta', 'PracticePace', 'IsWet', 'DriverConsistency', 'TrackPerformance', 'TeamAvgPoints', 'ReliabilityRisk']
            dummy_row = np.array([[
                1.0, float(current_grid), 0.0, 300.0, 1.0, 
                float(q_delta), float(p_pace), float(is_wet),
                float(driver_cons), float(track_perf), float(team_avg), float(reliability)
            ]])
            
            # FIX: Use DataFrame to suppress warning
            dummy_df = pd.DataFrame(dummy_row, columns=scaler_cols)
            scaled_curr = self.scaler.transform(dummy_df)[0]
            
            # Extract Transformed Values for Network inputs [Grid, Quali, Pace, Wet, DriverCons, TrackPerf, TeamAvg, Reliability]
            curr_grid_s = scaled_curr[1]
            curr_quali_s = scaled_curr[5]
            curr_pace_s = scaled_curr[6]
            curr_wet_s = scaled_curr[7]
            curr_cons_s = scaled_curr[8]
            curr_trackperf_s = scaled_curr[9]
            curr_team_avg_s = scaled_curr[10]
            curr_reliability_s = scaled_curr[11]
            
            curr_feat = torch.tensor([[curr_grid_s, curr_quali_s, curr_pace_s, curr_wet_s, curr_cons_s, curr_trackperf_s, curr_team_avg_s, curr_reliability_s]], dtype=torch.float32).to(self.device)
            curr_team = torch.tensor([t_id], dtype=torch.long).to(self.device)
            curr_track = torch.tensor([tr_id], dtype=torch.long).to(self.device)
            
            # 3. Predict
            with torch.no_grad():
                pred = self.model(hist_num, curr_feat, hist_team, hist_track, curr_team, curr_track)
                return float(pred.item())
                
        except Exception as e:
            logger.error(f"LSTM Predict Error: {e}")
            return None

    def get_correction_factor(self, driver, s_pos, team, track, year, round_num, sim_rank, context={}, track_lap_time=80.0):
        pred = self.predict_position(driver, year, round_num, s_pos, team, track, context)
        if pred is None: return 0.0
        
        diff = pred - sim_rank
        # Dynamic scaling: 1 position delta ~ 5% of track lap time (e.g. 4.5s on a 90s lap)
        scale = track_lap_time * 0.05
        return np.clip(diff * scale, -scale * 10, scale * 10)
