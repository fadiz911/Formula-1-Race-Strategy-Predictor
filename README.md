# 🏁 Formula 1 Race Strategy Predictor & Simulator

**A Multi-Model Deep Learning Platform & Multi-Agent AI System for F1 Finishing Position Prediction, Strategy Optimization, & Race Simulation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![FastF1](https://img.shields.io/badge/FastF1-Latest-green)](https://github.com/theOehrly/Fast-F1)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-Ready-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An advanced deep learning framework, LangGraph-powered multi-agent AI system, and Monte Carlo simulation engine that predicts Formula 1 finishing positions, race times, pit-stop strategies, and reliability risks using live and historical FastF1 session telemetry.

---

## 📊 Comprehensive Metrics & Performance

The platform employs a multi-tiered metrics framework to evaluate deep learning predictions, physics simulations, and strategic recommendations against official FIA timing data:

### 1. Model Ranking & Accuracy Metrics

| Metric | Target / Formula | Model Performance | Primary Focus |
| :--- | :--- | :--- | :--- |
| **Spearman Rank Correlation ($\rho$)** | Order matching across grid | **0.988** (Ensemble)<br>**0.987** (LSTM V4)<br>**0.967** (Transformer V5) | Evaluates how accurately the driver finishing order matches actual FIA results. |
| **Pearson Correlation ($r$)** | Linear position alignment | **0.95+** (Clean Correlation) | Measures linear association between predicted and actual positions across full seasons. |
| **Mean Absolute Error (MAE)** | $\frac{1}{N} \sum \|y_i - \hat{y}_i\|$ | **0.83 positions** (LSTM)<br>**1.09 positions** (Transformer) | Tracks average positional deviation across all drivers. |
| **Absolute Misplacement** | $\| \text{Actual} - \text{Predicted} \|$ | Per-driver error delta | Flags individual driver placement drift per race session. |

### 2. Diagnostic & Evaluation Metrics

* **Wet vs. Dry MAE Breakdown**: Tracks weather sensitivity by separating error distribution between dry sessions and rainy/wet flag sessions (`analyze_errors.py`).
* **Clean vs. Raw Correlation**: Differentiates standard racing conditions (*Clean Correlation*, finish $< \text{P18}$) from stochastic DNF/collision events (*Raw Correlation* in `src/benchmark_suite.py`).
* **Podium Hits / Podium Accuracy**: Evaluates top-3 set intersection between predicted podium finishers and actual FIA classified podium drivers (`src/evaluator.py`).
* **Driver Volatility Index**: Identifies drivers with high performance variance or mechanical retirement risks.

### 3. Simulation & Telemetry Metrics

* **Tire Degradation Delta**: Lap-time penalty curves per compound (`SOFT`, `MEDIUM`, `HARD`) over stint lengths (`src/tire_degradation_tool.py`).
* **Qualifying Pace Delta (`QualiDelta`)**: Normalized lap time differential relative to teammate and grid average.
* **Practice Pace Delta (`PracticePace`)**: Long-run race pace profile extracted from FP2/FP3 telemetry.
* **Grid-to-Finish Position Shift**: Net position gain/loss ($\Delta = \text{Grid Position} - \text{Predicted Finish}$).

---

## 🧠 Multi-Model & Multi-Agent Architecture

### 🤖 LangGraph Multi-Agent Strategy System (`multi_agent_app.py`)
A collaborative AI agent network built with LangGraph to orchestrate race strategy recommendations:

```
                  ┌───────────────────────────┐
                  │   User Session Context    │
                  └─────────────┬─────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │    Telemetry Analyst      │
                  │   (Lap Pace & Sector Δ)   │
                  └─────────────┬─────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │    Reliability Auditor    │
                  │   (DNF Risk & History)    │
                  └─────────────┬─────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │    Race Strategist        │
                  │ (Stint & Pit Optimizer)   │
                  └─────────────┬─────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │   Chief Strategy Director │
                  │  (Final Recommendation)  │
                  └───────────────────────────┘
```

* **Telemetry Analyst Agent**: Processes sector times, speed trap data, and telemetry deltas.
* **Reliability Auditor Agent**: Evaluates DNF probabilities, power unit wear, and track crash history.
* **Race Strategist Agent**: Calculates optimal pit-stop windows (1-stop vs 2-stop) and compound wear.
* **Chief Strategy Director Agent**: Synthesizes agent inputs into actionable pit windows and risk-adjusted race plans.

---

### 🔮 Deep Learning Predictor Engines

#### 1. Ensemble Predictor (Max Accuracy)
Integrates predictions from the LSTM and Transformer networks:
- **70% Weight**: LSTM Sequence model (ensures temporal ranking stability).
- **30% Weight**: Transformer model (captures pace spikes and reliability deviations).

#### 2. LSTM Neural Network (V4)
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

#### 3. Transformer Neural Network (V5)
Applies Multi-Head Self-Attention over sequence context, driver-to-teammate pace differentials, and track-specific history:
- **Telemetry-Driven**: Leverages practice lap time profiles and qualifying deltas.
- **Reliability-Aware**: Blends reliability risks (DNF probability) into sorting weights.

---

## 🎲 Monte Carlo Simulation Engine

Simulate full Grand Prix runs on a lap-by-lap basis (`src/simulation.py`):
- **Tire Degradation**: Interactive calibration of `SOFT`, `MEDIUM`, and `HARD` compound degradation curves.
- **Race Context**: Safety Cars (SC), Virtual Safety Cars (VSC), and dynamic rain/wet flag conditions.
- **Form Index**: Dynamic driver form adjustments based on recent results.
- **Starting Grid Overrides**: Customize starting positions with automatic support for grid sizes above 20 to handle grid penalties or substitute drivers.

---

## 📁 Comprehensive Project Structure

```
Formula-1-Race-Strategy-Predictor/
├── src/
│   ├── lstm_model.py                # PyTorch LSTM neural network definition
│   ├── lstm_predictor.py            # LSTM inference and sequence pipeline
│   ├── transformer_model.py         # PyTorch Transformer neural network definition
│   ├── transformer_predictor.py     # Transformer inference pipeline
│   ├── ensemble_predictor.py        # Blended Ensemble model predictor
│   ├── multi_agent_graph.py         # LangGraph state graph and multi-agent workflow
│   ├── strategist_agent.py          # Race Strategist agent logic
│   ├── benchmark_suite.py           # Pearson correlation & validation benchmark suite
│   ├── evaluator.py                 # Automated FIA timing auditor & podium accuracy check
│   ├── reliability_model.py         # DNF probability and reliability risk calculation
│   ├── telemetry.py                 # FastF1 telemetry loader & sector analysis
│   ├── tire_degradation_tool.py     # Compound degradation curve modeling
│   ├── strategy_engine.py           # Pace deltas and race time calculations
│   ├── optimizer.py                 # Pit-stop strategy optimizer (1-stop vs 2-stop)
│   ├── simulation.py                # Lap-by-lap Monte Carlo simulation engine
│   ├── feature_engineering.py       # Telemetry extraction & temporal feature scaling
│   ├── train_lstm_enhanced.py       # LSTM training pipeline
│   └── train_transformer_enhanced.py# Transformer training pipeline
├── models_dl/                       # Saved LSTM PyTorch weights & scalers
├── models_transformer/              # Saved Transformer PyTorch weights & scalers
├── reports/                         # Generated evaluation summaries & Markdown reports
├── app.py                           # Single-agent & Simulation Streamlit UI
├── multi_agent_app.py               # LangGraph Multi-Agent Streamlit UI
├── simulate_agent.py                # Command-line agent simulation interface
├── evaluate_all_predictors.py       # Spearman rank correlation evaluation (2024–2025)
├── analyze_errors.py                # Error diagnostics (Wet vs Dry, hardest drivers)
├── analyze_season.py                # Season-wide analysis script
├── generate_report.py               # Automated Markdown race report generator
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## 🚀 Installation & Usage

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/fadiz911/Formula-1-Race-Strategy-Predictor.git
cd Formula-1-Race-Strategy-Predictor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Web Interfaces

* **Standard Race Predictor & Simulator App**:
  ```bash
  streamlit run app.py
  ```

* **Multi-Agent LangGraph AI Strategy App**:
  ```bash
  streamlit run multi_agent_app.py
  ```

### 3. Run Agent CLI Simulation
```bash
python simulate_agent.py
```

### 4. Run Evaluation & Diagnostic Suites
```bash
# Evaluate Spearman correlations across 2024–2025 seasons
python evaluate_all_predictors.py

# Run wet vs. dry weather diagnostics and error outlier analysis
python analyze_errors.py

# Run benchmark suite (Pearson Raw vs Clean correlation)
python -m src.benchmark_suite
```

---

## 📄 License & Contact
- **License**: MIT License
- **Author**: Fadi Zoabi ([@fadiz911](https://github.com/fadiz911))