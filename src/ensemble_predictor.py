import numpy as np
import logging
from src.lstm_predictor import LSTMPredictor
from src.transformer_predictor import TransformerPredictor

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self, lstm_weight=0.7, trans_weight=0.3):
        self.lstm = LSTMPredictor()
        self.transformer = TransformerPredictor()
        
        self.lstm_weight = lstm_weight
        self.trans_weight = trans_weight
        
        self.loaded = self.lstm.loaded and self.transformer.loaded
        
        if self.loaded:
            logger.info(f"✅ Ensemble Predictor loaded (LSTM weight: {lstm_weight}, Transformer weight: {trans_weight})")
        else:
            logger.warning(f"⚠️ Ensemble Predictor partial load: LSTM={self.lstm.loaded}, Trans={self.transformer.loaded}")
            # If at least one is loaded, we can still run
            self.loaded = self.lstm.loaded or self.transformer.loaded
            
        # Expose raw_history_df from LSTM for Streamlit compatibility
        if self.lstm.loaded:
            self.raw_history_df = self.lstm.raw_history_df
            self.history_df = self.lstm.history_df
            self.scaler = self.lstm.scaler
        elif self.transformer.loaded:
            self.raw_history_df = self.transformer.raw_history_df
            self.history_df = self.transformer.history_df
            self.scaler = self.transformer.scaler

    def predict_position(self, driver, year, round_num, current_grid, current_team_name, current_track_name, context={}):
        lstm_scaled = None
        trans_scaled = None
        
        # 1. Get LSTM Prediction (Scaled)
        if self.lstm.loaded:
            lstm_scaled = self.lstm.predict_position(
                driver, year, round_num, current_grid, current_team_name, current_track_name, context
            )
                
        # 2. Get Transformer Prediction (Scaled by Transformer scaler)
        if self.transformer.loaded:
            trans_scaled_own = self.transformer.predict_position(
                driver, year, round_num, current_grid, current_team_name, current_track_name, context
            )
            if trans_scaled_own is not None:
                trans_unscaled = (trans_scaled_own * self.transformer.scaler.scale_[0]) + self.transformer.scaler.mean_[0]
                if self.lstm.loaded:
                    # Scale it using LSTM's scaler to match LSTM interface
                    trans_scaled = (trans_unscaled - self.lstm.scaler.mean_[0]) / self.lstm.scaler.scale_[0]
                else:
                    trans_scaled = trans_scaled_own
            
        # 3. Ensemble Blend
        if lstm_scaled is not None and trans_scaled is not None:
            blend_scaled = (self.lstm_weight * lstm_scaled) + (self.trans_weight * trans_scaled)
            return blend_scaled
        elif lstm_scaled is not None:
            return lstm_scaled
        elif trans_scaled is not None:
            return trans_scaled
        return None

    def get_correction_factor(self, driver, s_pos, team, track, year, round_num, sim_rank, context={}, track_lap_time=80.0):
        pred_scaled = self.predict_position(driver, year, round_num, s_pos, team, track, context)
        if pred_scaled is None: return 0.0
        
        # Unscale to get predicted position
        pred_pos = (pred_scaled * self.lstm.scaler.scale_[0]) + self.lstm.scaler.mean_[0]
        
        diff = pred_pos - sim_rank
        scale = track_lap_time * 0.05
        return np.clip(diff * scale, -scale * 10, scale * 10)
