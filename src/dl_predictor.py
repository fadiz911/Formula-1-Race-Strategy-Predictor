
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class RacePredictorNet(nn.Module):
    # Must match training definition
    def __init__(self, num_teams, num_tracks, embedding_dim=8):
        super(RacePredictorNet, self).__init__()
        input_dim = (embedding_dim * 2) + 4 
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, team_idx, track_idx, numerical_features):
        x = torch.cat([self.team_embedding(team_idx), self.track_embedding(track_idx), numerical_features], dim=1)
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.output(x)
        return x

class DLPredictor:
    def __init__(self, model_dir='models_dl'):
        self.model_path = os.path.join(model_dir, 'model.pth')
        self.encoders_path = os.path.join(model_dir, 'encoders.pkl')
        self.model = None
        self.encoders = None
        self.loaded = False
        self._load_model()
        
    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.warning("DL Model not found.")
                return

            with open(self.encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
                
            self.model = RacePredictorNet(self.encoders['num_teams'], self.encoders['num_tracks'])
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            self.loaded = True
            logger.info("✅ Deep Learning Predictor loaded.")
            
        except Exception as e:
            logger.error(f"Failed to load DL model: {e}")
            self.loaded = False

    def predict_position(self, grid_pos, team_name, track_name, year, round_num, rolling_points):
        if not self.loaded: return None
        try:
            # Encoders
            try: t_id = self.encoders['team'].transform([team_name])[0]
            except: t_id = 0 # Fallback
            
            try: tr_id = self.encoders['track'].transform([track_name])[0]
            except: tr_id = 0
            
            # Scaler
            # ['GridPos', 'RollingPoints', 'Year', 'Round']
            input_feats = pd.DataFrame([{
                'GridPos': grid_pos, 
                'RollingPoints': rolling_points, 
                'Year': year, 
                'Round': round_num
            }])
            scaled_feats = self.encoders['scaler'].transform(input_feats)
            
            # Tensorize
            t_team = torch.tensor([t_id], dtype=torch.long)
            t_track = torch.tensor([tr_id], dtype=torch.long)
            t_num = torch.tensor(scaled_feats, dtype=torch.float32)
            
            with torch.no_grad():
                pred = self.model(t_team, t_track, t_num)
                return float(pred.item())
                
        except Exception as e:
            # logger.warning(f"DL Inferences failed: {e}")
            return None

    def get_correction_factor(self, driver, grid_pos, team, track, year, round_num, rolling_points, sim_rank):
        pred = self.predict_position(grid_pos, team, track, year, round_num, rolling_points)
        if pred is None: return 0.0
        
        # DL predicts P5, Sim says P10 -> Delta = -5 (Speed up)
        # Actually my previous logic was: ml - sim.
        # P5 - P10 = -5. Negative means "Expected Better".
        # If Sim is P10 (slower) and ML is P5 (faster), we need to SUBTRACT time.
        
        pos_delta = pred - sim_rank
        # Heuristic: 1 pos = 2.5s
        correction = pos_delta * 2.5
        return np.clip(correction, -25.0, 25.0)
