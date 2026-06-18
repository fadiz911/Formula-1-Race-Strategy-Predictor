# 🏎️ F1 Deep Learning Predictor: Full Accuracy & Error Report

This report breaks down the performance of the **LSTM (V4)** and **Transformer (V5)** models across all analyzed 2024 and 2025 races. Both models are evaluated using **Spearman Rank Correlation** (how well they predict correct standing order) and **Mean Absolute Error (MAE)** in grid positions.

## 📊 Overall Performance Summary

| Metric | LSTM Predictor | Transformer Predictor |
| :--- | :---: | :---: |
| **2024 Avg Spearman Correlation** | 0.9780 | 0.9737 |
| **2025 Avg Spearman Correlation** | 0.9868 | 0.9673 |
| **Overall Mean Absolute Error (MAE)** | 0.83 pos | 1.09 pos |
| **Dry Weather MAE** | 0.82 pos | 1.09 pos |
| **Wet Weather MAE** | 0.86 pos | 1.10 pos |

---

## 🏆 Top 5 Most Accurate Races (Transformer)

These races achieved the highest rank correlation, indicating near-perfect prediction of the standing order.

| Year | Round | Race | Transformer Correlation | LSTM Correlation | Drivers |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 2024 | 14 | Belgian Grand Prix | **0.9941** | 0.9941 | 16 |
| 2024 | 7 | Emilia Romagna Grand Prix | **0.9912** | 0.9941 | 16 |
| 2024 | 10 | Spanish Grand Prix | **0.9912** | 0.9971 | 16 |
| 2025 | 16 | Italian Grand Prix | **0.9875** | 0.9993 | 16 |
| 2024 | 18 | Singapore Grand Prix | **0.9857** | 0.9893 | 15 |

---

## ⚠️ Top 5 Least Accurate Races (Transformer)

These races had the lowest Spearman correlation, representing unexpected race results or model misses.

| Year | Round | Race | Transformer Correlation | LSTM Correlation | Drivers |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 2025 | 7 | Emilia Romagna Grand Prix | **0.9106** | 0.9451 | 13 |
| 2025 | 9 | Spanish Grand Prix | **0.9257** | 0.9934 | 16 |
| 2024 | 20 | Mexico City Grand Prix | **0.9338** | 0.9912 | 14 |
| 2024 | 24 | Abu Dhabi Grand Prix | **0.9419** | 0.9728 | 16 |
| 2024 | 3 | Australian Grand Prix | **0.9433** | 0.9478 | 16 |

---

## 🔍 Deep Dive: Why Did the Model Miss?

### 📍 2025 Emilia Romagna Grand Prix (Trans Correlation: 0.9106)

Below are the drivers in this race sorted by largest Transformer prediction errors:

| Driver | Team | Grid | Actual | Predicted | Trans Error | DNF Risk |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| ALB | Williams | 7 | 5 | 7.3 | **2.27** | 0.00 |
| HAM | Ferrari | 12 | 4 | 6.1 | **2.11** | 0.00 |
| PIA | McLaren | 1 | 3 | 1.0 | **2.00** | 0.00 |
| RUS | Mercedes | 3 | 7 | 5.1 | **1.89** | 0.00 |
| STR | Aston Martin | 8 | 15 | 13.6 | **1.44** | 0.20 |

**Key factors contributing to the miss:**

- **Weather condition**: Dry race.
- **Grid Climbers**: **HAM** made massive grid recoveries from the back of the pack, which statistical sequence history did not fully anticipate.


### 📍 2025 Spanish Grand Prix (Trans Correlation: 0.9257)

Below are the drivers in this race sorted by largest Transformer prediction errors:

| Driver | Team | Grid | Actual | Predicted | Trans Error | DNF Risk |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| HUL | Kick Sauber | 15 | 5 | 10.1 | **5.12** | 0.20 |
| ANT | Mercedes | 6 | 20 | 16.4 | **3.60** | 0.00 |
| VER | Red Bull Racing | 3 | 10 | 7.2 | **2.83** | 0.00 |
| OCO | Haas F1 Team | 16 | 16 | 13.2 | **2.81** | 0.20 |
| LEC | Ferrari | 7 | 3 | 4.9 | **1.91** | 0.00 |

**Key factors contributing to the miss:**

- **Weather condition**: Dry race.
- **Grid Climbers**: **HUL** made massive grid recoveries from the back of the pack, which statistical sequence history did not fully anticipate.


### 📍 2024 Mexico City Grand Prix (Trans Correlation: 0.9338)

Below are the drivers in this race sorted by largest Transformer prediction errors:

| Driver | Team | Grid | Actual | Predicted | Trans Error | DNF Risk |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| MAG | Haas F1 Team | 7 | 7 | 10.9 | **3.85** | 0.00 |
| ALO | Aston Martin | 13 | 20 | 17.2 | **2.85** | 0.00 |
| RUS | Mercedes | 5 | 5 | 2.2 | **2.81** | 0.20 |
| ALB | Williams | 9 | 20 | 17.8 | **2.17** | 0.20 |
| SAI | Ferrari | 1 | 1 | 2.9 | **1.90** | 0.00 |

**Key factors contributing to the miss:**

- **Weather condition**: Dry race.


### 📍 2024 Abu Dhabi Grand Prix (Trans Correlation: 0.9419)

Below are the drivers in this race sorted by largest Transformer prediction errors:

| Driver | Team | Grid | Actual | Predicted | Trans Error | DNF Risk |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| HAM | Mercedes | 16 | 4 | 8.3 | **4.26** | 0.00 |
| BOT | Kick Sauber | 9 | 20 | 16.2 | **3.84** | 0.00 |
| PIA | McLaren | 2 | 10 | 7.6 | **2.36** | 0.00 |
| STR | Aston Martin | 13 | 14 | 11.7 | **2.33** | 0.00 |
| RUS | Mercedes | 6 | 5 | 3.2 | **1.82** | 0.00 |

**Key factors contributing to the miss:**

- **Weather condition**: Dry race.
- **Grid Climbers**: **HAM** made massive grid recoveries from the back of the pack, which statistical sequence history did not fully anticipate.


### 📍 2024 Australian Grand Prix (Trans Correlation: 0.9433)

Below are the drivers in this race sorted by largest Transformer prediction errors:

| Driver | Team | Grid | Actual | Predicted | Trans Error | DNF Risk |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| VER | Red Bull Racing | 1 | 20 | 16.1 | **3.94** | 0.00 |
| HAM | Mercedes | 11 | 20 | 16.6 | **3.38** | 0.00 |
| LEC | Ferrari | 4 | 2 | 4.6 | **2.58** | 0.00 |
| SAI | Ferrari | 2 | 1 | 3.3 | **2.31** | 0.00 |
| HUL | Haas F1 Team | 16 | 9 | 10.6 | **1.62** | 0.00 |

**Key factors contributing to the miss:**

- **Weather condition**: Dry race.



---

## 📋 Complete Race-by-Race Standing Report

| Year | Round | Race | Drivers | LSTM Corr | Trans Corr | LSTM MAE | Trans MAE |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 2024 | 1 | Bahrain Grand Prix | 16 | 0.9235 | 0.9618 | 1.27 | 1.11 |
| 2024 | 2 | Saudi Arabian Grand Prix | 15 | 0.9714 | 0.9821 | 1.30 | 1.00 |
| 2024 | 3 | Australian Grand Prix | 16 | 0.9478 | 0.9433 | 1.09 | 1.34 |
| 2024 | 4 | Japanese Grand Prix | 16 | 0.9875 | 0.9728 | 0.87 | 0.88 |
| 2024 | 7 | Emilia Romagna Grand Prix | 16 | 0.9941 | 0.9912 | 0.80 | 0.69 |
| 2024 | 8 | Monaco Grand Prix | 16 | 0.9904 | 0.9713 | 0.80 | 1.05 |
| 2024 | 9 | Canadian Grand Prix | 16 | 0.9792 | 0.9792 | 1.32 | 1.47 |
| 2024 | 10 | Spanish Grand Prix | 16 | 0.9971 | 0.9912 | 0.47 | 0.44 |
| 2024 | 12 | British Grand Prix | 16 | 0.9853 | 0.9765 | 0.85 | 1.12 |
| 2024 | 13 | Hungarian Grand Prix | 16 | 0.9529 | 0.9824 | 0.91 | 0.76 |
| 2024 | 14 | Belgian Grand Prix | 16 | 0.9941 | 0.9941 | 0.55 | 0.61 |
| 2024 | 15 | Dutch Grand Prix | 16 | 0.9824 | 0.9824 | 0.59 | 0.83 |
| 2024 | 16 | Italian Grand Prix | 16 | 0.9912 | 0.9824 | 0.54 | 0.93 |
| 2024 | 17 | Azerbaijan Grand Prix | 14 | 0.9956 | 0.9692 | 0.89 | 1.22 |
| 2024 | 18 | Singapore Grand Prix | 15 | 0.9893 | 0.9857 | 0.84 | 1.21 |
| 2024 | 20 | Mexico City Grand Prix | 14 | 0.9912 | 0.9338 | 0.94 | 1.39 |
| 2024 | 22 | Las Vegas Grand Prix | 16 | 0.9588 | 0.9853 | 0.98 | 0.77 |
| 2024 | 24 | Abu Dhabi Grand Prix | 16 | 0.9728 | 0.9419 | 0.98 | 1.43 |
| 2025 | 1 | Australian Grand Prix | 12 | 0.9912 | 0.9632 | 1.18 | 1.58 |
| 2025 | 3 | Japanese Grand Prix | 13 | 0.9890 | 0.9725 | 0.38 | 0.74 |
| 2025 | 4 | Bahrain Grand Prix | 13 | 0.9821 | 0.9711 | 0.80 | 1.61 |
| 2025 | 5 | Saudi Arabian Grand Prix | 14 | 0.9857 | 0.9769 | 0.95 | 0.94 |
| 2025 | 7 | Emilia Romagna Grand Prix | 13 | 0.9451 | 0.9106 | 0.82 | 1.18 |
| 2025 | 8 | Monaco Grand Prix | 16 | 0.9853 | 0.9706 | 0.65 | 1.21 |
| 2025 | 9 | Spanish Grand Prix | 16 | 0.9934 | 0.9257 | 0.60 | 1.67 |
| 2025 | 10 | Canadian Grand Prix | 16 | 0.9912 | 0.9765 | 0.65 | 1.03 |
| 2025 | 11 | Austrian Grand Prix | 16 | 0.9941 | 0.9853 | 0.71 | 0.97 |
| 2025 | 12 | British Grand Prix | 16 | 0.9867 | 0.9837 | 1.24 | 1.37 |
| 2025 | 14 | Hungarian Grand Prix | 16 | 0.9824 | 0.9853 | 0.74 | 0.76 |
| 2025 | 15 | Dutch Grand Prix | 16 | 0.9993 | 0.9757 | 0.63 | 1.35 |
| 2025 | 16 | Italian Grand Prix | 16 | 0.9993 | 0.9875 | 0.52 | 0.59 |
| 2025 | 17 | Azerbaijan Grand Prix | 16 | 0.9941 | 0.9441 | 0.65 | 1.21 |
| 2025 | 18 | Singapore Grand Prix | 16 | 1.0000 | 0.9853 | 0.38 | 0.78 |
| 2025 | 20 | Mexico City Grand Prix | 16 | 0.9764 | 0.9499 | 1.02 | 1.77 |
| 2025 | 22 | Las Vegas Grand Prix | 16 | 0.9808 | 0.9808 | 1.18 | 1.35 |