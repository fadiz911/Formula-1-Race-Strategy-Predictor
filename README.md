# 🏁 Formula 1 Race Strategy Predictor & Simulator

**A Multi-Model Deep Learning Platform for F1 Finishing Position Prediction & Race Simulation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![FastF1](https://img.shields.io/badge/FastF1-Latest-green)](https://github.com/theOehrly/Fast-F1)
[![Streamlit](https://img.shields.io/badge/Streamlit-Ready-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An advanced deep learning framework and Monte Carlo simulation engine that predicts Formula 1 finishing positions, race times, and pit-stop strategies using actual session telemetry. Features a high-accuracy Ensemble model that blends **LSTM (V4)** and **Transformer (V5)** architectures.

---

## 📊 Model Performance

### Spearman Rank Correlation (2024–2025 Seasons)
Evaluated on how accurately the models predict the correct order of finishing drivers.

| Model / Architecture | 2024 Avg Correlation | 2025 Avg Correlation | Primary Focus |
| :--- | :---: | :---: | :--- |
| **Ensemble (70% LSTM / 30% Trans)** | **0.982** | **0.988** | Max accuracy & sequence stability |
| **LSTM (V4) - Temporal Sequence** | **0.978** | **0.987** | Rolling form & consistency tracking |
| **Transformer (V5) - Telemetry Enabled** | **0.974** | **0.967** | Driver pace delta & reliability profiling |

### Mean Absolute Error (MAE)
- **LSTM (V4) Overall MAE**: **0.83 positions** (average error under 1 position)
- **Transformer (V5) Overall MAE**: **1.09 positions**

---

## 🧠 Multi-Model Architecture

The application hosts three prediction architectures, allowing users to toggle between different methods:

### 1. Ensemble Predictor (Max Accuracy)
Integrates predictions from the LSTM and Transformer networks:
- **70% Weight**: LSTM Sequence model (ensures smooth ranking transitions).
- **30% Weight**: Transformer model (captures pace spikes and reliability deviations).

### 2. LSTM Neural Network (V4)
Captures chronological form and momentum across sliding 5-race sequence windows:
```
Input Sequence (5 historical races x 10 features)
    ↓
LSTM Layer (128 hidden units, 2 layers)
    ↓
Attention Pooling (time-weighted performance weighting)
    ↓
Dense Layers (32 → 1) + Current Context Injection
    ↓
Predicted Finishing Position
```

### 3. Transformer Neural Network (V5)
Applies Multi-Head Attention over sequence context, driver-to-teammate pace differentials, and track-specific history:
- **Telemetry-Driven**: Leverages practice lap time profiles and qualifying deltas.
- **Reliability-Aware**: Blends reliability risks (DNF probability) into the sorting weights.

---

## 🎲 Monte Carlo Simulation Engine

Simulate full Grand Prix runs on a lap-by-lap basis. The simulator models:
- **Tire Degradation**: Interactive calibration of SOFT, MEDIUM, and HARD compound degradation curves.
- **Race Context**: Safety Cars, Virtual Safety Cars, and weather/wet flag conditions.
- **Form Index**: Dynamic driver form adjustments based on recent results.
- **Starting Grid Overrides**: Customize starting positions (with automatic support for grid sizes above 20 to handle grid penalties or substitute drivers).

---

## 📁 Project Structure

```
Formula-1-Race-Strategy-Predictor/
├── src/
│   ├── lstm_model.py                # LSTM neural network design
│   ├── lstm_predictor.py            # LSTM prediction engine
│   ├── transformer_model.py         # Transformer neural network design
│   ├── transformer_predictor.py     # Transformer prediction engine
│   ├── ensemble_predictor.py        # Blended predictor assembly
│   ├── train_lstm_enhanced.py       # LSTM training routine
│   ├── train_transformer_enhanced.py# Transformer training routine
│   ├── strategy_engine.py           # Field pace and delta calculation
│   ├── optimizer.py                 # Tire strategy optimizer (1-stop vs 2-stop)
│   ├── simulation.py                # Monte Carlo lap-by-lap simulator
│   └── feature_engineering.py       # Telemetry and history feature extraction
├── models_dl/
│   ├── lstm_model_v3.pth            # Pre-trained LSTM weights
│   └── lstm_artifacts_v3.pkl        # Scaler parameters
├── models_transformer/
│   ├── transformer_model.pth        # Pre-trained Transformer weights
│   └── artifacts.pkl                # Scaler parameters
├── reports/
│   ├── predictor_comparison_summary.csv # Raw validation stats
│   └── race_prediction_report.md    # Detailed race breakdown & errors
├── app.py                           # Premium Streamlit web app
└── requirements.txt                 # Dependencies
```

---

## 🚀 Installation & Usage

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/fadiz911/Formula-1-Race-Strategy-Predictor.git
cd Formula-1-Race-Strategy-Predictor

# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Web UI
```bash
streamlit run app.py
```

### 3. Evaluate Predictors
```bash
python evaluate_all_predictors.py
python analyze_errors.py
```

---

## 📄 License & Contact
- **License**: MIT License
- **Author**: Fadi Zoabi ([@fadiz911](https://github.com/fadiz911))