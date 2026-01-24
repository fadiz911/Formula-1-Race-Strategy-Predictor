# 🏁 Formula 1 Race Strategy Predictor

**Deep Learning Model for F1 Finishing Position Prediction**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![FastF1](https://img.shields.io/badge/FastF1-Latest-green)](https://github.com/theOehrly/Fast-F1)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An advanced LSTM-based neural network that predicts Formula 1 race finishing positions with **97% correlation accuracy** and **sub-1 position error**.

---

## 🎯 Model Performance

### Global Results (2023-2025 Seasons)
| Metric | Score | Details |
|--------|-------|---------|
| **Pearson Correlation** | **0.9707** | 97% prediction accuracy |
| **Clean Correlation** | **0.9655** | 96.5% (excluding DNFs) |
| **Mean Absolute Error** | **0.91 positions** | Average error < 1 position |
| **Total Predictions** | 982 races | Across 3 complete seasons |

### Per-Season Performance
- **2024 Season**: 0.99 correlation (near-perfect), 0.6-0.8 MAE
- **2025 Season**: 0.99 correlation, consistently sub-1.0 MAE
- **Best Races**: Canadian GP 2024 (0.99 corr, 0.6 MAE), Japanese GP 2025 (1.00 corr, 0.5 MAE)

---

## 🧠 Model Architecture

### LSTM Neural Network (V4)
The model uses a Long Short-Term Memory (LSTM) architecture designed to capture temporal dependencies in racing performance.

```
Input Layer (10 Features)
    ↓
LSTM Layer (128 hidden units, 2 layers)
    ↓
Attention Mechanism (Historical Race Weighting)
    ↓
Current Race Context (6 Features)
    ↓
Fully Connected Layers (32 → 1)
    ↓
Predicted Finishing Position
```

### Input Features (10 Total)

#### Historical Sequence Features (Past 5 Races)
1. **FinishPos** - Previous finishing positions
2. **GridPos** - Starting grid positions
3. **Points** - Championship points earned
4. **SpeedST** - Speed trap velocity
5. **StintCount** - Number of pit stops
6. **QualiDelta** - Qualifying vs teammate gap
7. **PracticePace** - Practice session performance
8. **IsWet** - Weather conditions (0=dry, 1=wet)
9. **DriverConsistency** - Rolling std dev of last 5 finishes
10. **TrackPerformance** - Driver's average finish at this track

#### Current Race Context (6 Features)
- Grid position (starting position)
- Qualifying delta (vs teammate)
- Practice pace differential
- Weather conditions
- Driver consistency score
- Track-specific performance

### Model Training
- **Loss Function**: Huber Loss (robust to DNF outliers)
- **Optimizer**: Adam (learning rate: 0.001)
- **Epochs**: 300
- **Scheduler**: ReduceLROnPlateau (adaptive learning)
- **Final Training Loss**: 0.026

---

## 📊 Sample Results

### 2024 Season Highlights
```
2024 Canadian GP:        0.99 correlation, 0.6 MAE
2024 Dutch GP:           0.99 correlation, 0.5 MAE
2024 Spanish GP:         0.99 correlation, 0.6 MAE
2024 Hungarian GP:       0.99 correlation, 0.7 MAE
2024 Emilia Romagna GP:  0.99 correlation, 0.8 MAE
```

### 2025 Season Highlights
```
2025 Japanese GP:        1.00 correlation, 0.5 MAE
2025 Monaco GP:          0.99 correlation, 0.6 MAE
2025 Singapore GP:       0.99 correlation, 0.6 MAE
2025 Canadian GP:        0.99 correlation, 0.6 MAE
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone repository
git clone https://github.com/fadiz911/Formula-1-Race-Strategy-Predictor.git
cd Formula-1-Race-Strategy-Predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Streamlit Web App
Run the interactive web application:
```bash
streamlit run app.py
```

Features:
- Select any 2023-2025 Grand Prix
- View predicted finishing positions
- Compare predictions vs actual results
- Visualize driver performance trends

### Training the Model
Retrain with updated data:
```bash
python -m src.train_lstm_enhanced
```

### Benchmarking
Validate model performance:
```bash
python -m src.benchmark_suite
```

---

## 📁 Project Structure

```
Formula-1-Race-Strategy-Predictor/
├── src/
│   ├── train_lstm_enhanced.py    # Model training pipeline
│   ├── lstm_model.py              # LSTM architecture
│   ├── lstm_predictor.py          # Inference engine
│   └── benchmark_suite.py         # Performance validation
├── models_dl/
│   ├── lstm_model_v3.pth          # Trained weights
│   └── lstm_artifacts_v3.pkl      # Scalers & encoders
├── app.py                         # Streamlit interface
└── requirements.txt               # Dependencies
```

---

## 🔬 Technical Details

### Data Pipeline
1. **Data Source**: FastF1 API (official F1 telemetry)
2. **Preprocessing**: StandardScaler normalization
3. **Feature Engineering**: Rolling statistics, track-specific metrics
4. **Sequence Generation**: 5-race sliding window

### Key Innovations
- **Phase 1**: Huber Loss + LR Scheduling (0.85 → baseline)
- **Phase 2**: Driver Consistency + Track Performance (0.97 → final)
- **Outlier Handling**: Robust loss function for crash/DNF scenarios
- **Temporal Modeling**: LSTM captures momentum and form trends

---

## 📈 Future Improvements

- [ ] Real-time prediction during live races
- [ ] Tire strategy optimization
- [ ] Weather forecast integration
- [ ] Pit stop timing predictions
- [ ] Safety car probability modeling

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **FastF1**: For comprehensive F1 telemetry data
- **PyTorch**: Deep learning framework
- **Streamlit**: Interactive web interface

---

## 📧 Contact

**Fadi Zoabi**  
GitHub: [@fadiz911](https://github.com/fadiz911)

---

*Developed with ❤️ for Formula 1 and Machine Learning*