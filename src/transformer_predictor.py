import torch
import numpy as np
import pandas as pd
import os
import pickle
import logging
from src.transformer_model import DriverTransformer
from src.train_lstm_enhanced import get_enhanced_race_data

logger = logging.getLogger(__name__)

class TransformerPredictor:
    def __init__(self, model_dir='models_transformer'):
        self.model_path = os.path.join(model_dir, 'transformer_model.pth')
        self.encoders_path = os.path.join(model_dir, 'artifacts.pkl')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.encoders = None
        self.scaler = None
        self.history_df = pd.DataFrame()
        self.loaded = False
        
        # Telemetry Medians (from models/medians.pkl if available)
        self.medians = {
            'Qualy_AvgSpeed': 221.8,
            'Qualy_MaxSpeed': 326.0,
            'Qualy_AvgThrottle': 72.78,
            'Qualy_FullThrottlePct': 53.63,
            'Qualy_AvgBrake': 0.168,
            'Qualy_AvgGear': 5.68,
            'Qualy_GearChanges': 36.0
        }
        
        # Team points mapping
        self.team_points_map = {
            'Red Bull Racing': 25.0, 'Mercedes': 15.0, 'Ferrari': 18.0, 'McLaren': 12.0,
            'Aston Martin': 10.0, 'Alpine': 6.0, 'Williams': 2.0, 'Haas F1 Team': 1.0,
            'Alfa Romeo': 1.0, 'AlphaTauri': 1.0, 'Kick Sauber': 0.5, 'RB': 2.0
        }
        
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(self.encoders_path):
            with open(self.encoders_path, 'rb') as f:
                data = pickle.load(f)
                self.encoders = data['encoders']
                self.scaler = data['scaler']
                
                # Load Model
                # num_teams = 12, num_tracks = 19
                self.model = DriverTransformer(
                    num_teams=len(self.encoders['team'].classes_),
                    num_tracks=len(self.encoders['track'].classes_),
                    embedding_dim=16,
                    d_model=32,
                    nhead=2,
                    num_layers=2
                ).to(self.device)
                
                if os.path.exists(self.model_path):
                    try:
                        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                        self.model.eval()
                        self.loaded = True
                    except Exception as e:
                        logger.error(f"Transformer load weights failed: {e}")
                        self.loaded = False
                    
                    # Pre-load history
                    try:
                        cache_path = os.path.join('cache', 'history_clean.pkl')
                        if os.path.exists(cache_path):
                            with open(cache_path, 'rb') as f:
                                self.history_df = pickle.load(f)
                        else:
                            self.history_df = get_enhanced_race_data(years=[2022, 2023, 2024, 2025, 2026])
                            if not self.history_df.empty:
                                os.makedirs('cache', exist_ok=True)
                                with open(cache_path, 'wb') as f:
                                    pickle.dump(self.history_df, f)
                    except Exception as e:
                        logger.error(f"Failed to load history for Transformer: {e}")
                        self.history_df = pd.DataFrame()
                        
                    if not self.history_df.empty:
                        self.history_df = self.history_df.sort_values(['Year', 'Round'])
                        
                        # Normalize teams
                        def normalize_team(t):
                            t = str(t)
                            if 'Alfa Romeo' in t: return 'Kick Sauber'
                            if 'AlphaTauri' in t: return 'RB'
                            if 'Aston Martin' in t: return 'Aston Martin'
                            return t
                        self.history_df['Team'] = self.history_df['Team'].apply(normalize_team)
                        
                        # Filter known
                        known_teams = set(self.encoders['team'].classes_)
                        self.history_df = self.history_df[self.history_df['Team'].isin(known_teams)]
                        known_tracks = set(self.encoders['track'].classes_)
                        self.history_df = self.history_df[self.history_df['Track'].isin(known_tracks)]
                        self.history_df['TeamId'] = self.encoders['team'].transform(self.history_df['Team'].astype(str))
                        self.history_df['TrackId'] = self.encoders['track'].transform(self.history_df['Track'].astype(str))
                        
                        # Scale the history dataframe
                        feature_cols = self.encoders.get('feature_cols', self.scaler.feature_names_in_)
                        for col in feature_cols:
                            if col not in self.history_df.columns:
                                if col == 'TeamAvgPoints':
                                    self.history_df[col] = self.history_df['Team'].map(self.team_points_map).fillna(5.0)
                                else:
                                    self.history_df[col] = self.medians.get(col, 0.0)
                        self.history_df[feature_cols] = self.scaler.transform(self.history_df[feature_cols])
                else:
                    logger.warning("Transformer weights not found.")
        else:
            logger.warning("Transformer artifacts not found.")

    @property
    def raw_history_df(self):
        if self.history_df.empty:
            return self.history_df
        df = self.history_df.copy()
        feature_cols = self.encoders.get('feature_cols', self.scaler.feature_names_in_)
        df[feature_cols] = self.scaler.inverse_transform(df[feature_cols])
        return df

    def predict_position(self, driver, year, round_num, current_grid, current_team_name, current_track_name, context={}):
        if not self.loaded or self.history_df.empty:
            return None
        
        try:
            # 1. Get History (Last 5 races) using unscaled years/rounds
            raw_hist = self.raw_history_df
            d_hist_raw = raw_hist[
                (raw_hist['Driver'] == driver) &
                (
                    (raw_hist['Year'] < year) |
                    ((raw_hist['Year'] == year) & (raw_hist['Round'] < round_num))
                )
            ].sort_values(['Year', 'Round'])
            
            if len(d_hist_raw) < 5:
                return None
            
            # Map back to scaled dataframe records via index
            indices = d_hist_raw.tail(5).index
            hist_recs = self.history_df.loc[indices].to_dict('records')
            
            # Construct Sequence features (10 features) - already scaled in constructor
            feature_cols = self.encoders.get('feature_cols', self.scaler.feature_names_in_)
            hist_vals_scaled = np.array([[float(x[col]) for col in feature_cols] for x in hist_recs])
            
            hist_num = torch.tensor(hist_vals_scaled[np.newaxis, :, :], dtype=torch.float32).to(self.device)
            hist_team = torch.tensor(np.array([[x['TeamId'] for x in hist_recs]]), dtype=torch.long).to(self.device)
            hist_track = torch.tensor(np.array([[x['TrackId'] for x in hist_recs]]), dtype=torch.long).to(self.device)
            
            # 2. Current features
            try:
                t_id = self.encoders['team'].transform([str(current_team_name)])[0]
            except:
                t_id = 0
            try:
                tr_id = self.encoders['track'].transform([str(current_track_name)])[0]
            except:
                tr_id = 0
                
            q_delta = context.get('QualiDelta', 0.0)
            p_pace = context.get('PracticePace', 0.0)
            is_wet = context.get('IsWet', 0.0)
            driver_cons = context.get('DriverConsistency', 5.0)
            track_perf = context.get('TrackPerformance', 10.5)
            team_avg = context.get('TeamAvgPoints', 5.0)
            reliability = context.get('ReliabilityRisk', 0.1)
            
            scaler_cols = ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 'QualiDelta', 'PracticePace', 'IsWet', 'DriverConsistency', 'TrackPerformance', 'TeamAvgPoints', 'ReliabilityRisk']
            dummy_row = np.array([[
                1.0, float(current_grid), 0.0, 300.0, 1.0, 
                float(q_delta), float(p_pace), float(is_wet),
                float(driver_cons), float(track_perf), float(team_avg), float(reliability)
            ]])
            
            dummy_df = pd.DataFrame(dummy_row, columns=scaler_cols)
            scaled_curr = self.scaler.transform(dummy_df)[0]
            
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
            logger.error(f"Transformer Predict Error: {e}")
            return None

    def get_correction_factor(self, driver, s_pos, team, track, year, round_num, sim_rank, context={}, track_lap_time=80.0):
        pred = self.predict_position(driver, year, round_num, s_pos, team, track, context)
        if pred is None: return 0.0
        
        diff = pred - sim_rank
        # Dynamic scaling: 1 position delta ~ 5% of track lap time (e.g. 4.5s on a 90s lap)
        scale = track_lap_time * 0.05
        return np.clip(diff * scale, -scale * 10, scale * 10)
