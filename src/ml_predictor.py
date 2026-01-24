
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self, model_dir='models'):
        self.model_path = os.path.join(model_dir, 'race_predictor.json')
        self.encoders_path = os.path.join(model_dir, 'encoders.pkl')
        self.medians_path = os.path.join(model_dir, 'medians.pkl')
        self.model = None
        self.encoders = None
        self.medians = None
        self.loaded = False
        self._load_model()
        
    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"ML Model not found at {self.model_path}. ML features will be disabled.")
                return

            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_path)
            
            with open(self.encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
            
            if os.path.exists(self.medians_path):
                with open(self.medians_path, 'rb') as f:
                    self.medians = pickle.load(f)
            else:
                self.medians = {} # Should probably warn
                
            self.loaded = True
            logger.info("✅ ML Predictor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.loaded = False

    def predict_position(self, grid_pos, team_name, track_name, year, telemetry_data=None):
        """
        Predict finishing position using ML model.
        Returns: predicted_pos (float) or None if model not loaded
        """
        if not self.loaded:
            return None
            
        try:
            # Prepare input vector
            # Features: ['GridPos', 'TeamAvgPoints', 'TrackId', 'TeamId', 'Year', 'Qualy_AvgSpeed'...]
            
            # encode team
            try:
                team_id = self.encoders['team'].transform([team_name])[0]
            except:
                team_id = 0 # Fallback for unknown team
            
            # encode track
            # fuzzy match or direct? simplified to direct for now
            try:
                track_id = self.encoders['track'].transform([track_name])[0]
            except:
                track_id = 0 # Fallback
            
            # team avg points (simplified lookup - in real prod would query DB)
            team_points_map = {
                'Red Bull Racing': 25, 'Mercedes': 15, 'Ferrari': 18, 'McLaren': 12,
                'Aston Martin': 10, 'Alpine': 6, 'Williams': 2, 'Haas F1 Team': 1,
                'Alfa Romeo': 1, 'AlphaTauri': 1, 'Kick Sauber': 0, 'RB': 2
            }
            team_avg_points = team_points_map.get(team_name, 5.0)

            input_dict = {
                'GridPos': grid_pos,
                'TeamAvgPoints': team_avg_points,
                'TrackId': track_id,
                'TeamId': team_id,
                'Year': year
            }
            
            # Add Telemetry
            tel_cols = ['Qualy_AvgSpeed', 'Qualy_MaxSpeed', 'Qualy_AvgThrottle', 'Qualy_FullThrottlePct', 'Qualy_AvgBrake', 'Qualy_GearChanges']
            
            if telemetry_data is None: telemetry_data = {}
            
            for col in tel_cols:
                # Use provided value -> median fallback -> 0 fallback
                val = telemetry_data.get(col)
                if val is None:
                    val = self.medians.get(col, 0.0)
                input_dict[col] = val

            input_data = pd.DataFrame([input_dict])
            
            # Reorder columns to match training EXACTLY
            # We need to know the order... 
            # Best practice: XGBoost handles named columns if using dataframe.
            
            pred = self.model.predict(input_data)[0]
            return float(pred)
            
        except Exception as e:
            logger.warning(f"ML Prediction failed: {e}")
            return None

    def get_correction_factor(self, driver, grid_pos, team, track, year, simulated_time_rank):
        """
        Returns a time adjustment (seconds) based on ML prediction vs Simulation.
        
        If ML predicts much worse position than Sim, we add time (positive).
        If ML predicts better, we subtract time (negative).
        """
        ml_pos = self.predict_position(grid_pos, team, track, year)
        
        if ml_pos is None:
            return 0.0
            
        # Delta: Positive means ML says we finish worse (higher number)
        # e.g. Sim says P5, ML says P10. Delta = 5 positions worse.
        # We should slow down the sim time.
        
        position_delta = ml_pos - simulated_time_rank
        
        # Heuristic: 1 position ~ 3 seconds gap on average?
        # We apply a soft correction
        time_penalty = position_delta * 2.5 
        
        # Clip to avoid extreme corrections
        return np.clip(time_penalty, -20.0, 20.0)
