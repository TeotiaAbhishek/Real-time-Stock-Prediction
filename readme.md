# 📈 Real-Time Stock Price Prediction Using Machine Learning

A real-time stock forecasting system that combines **traditional machine learning**, **deep learning**, and **time-series modelling** techniques to predict next-day stock movements and generate decision-support signals.

The project compares multiple forecasting approaches including:

- Random Forest
- XGBoost
- ARIMA
- LSTM
- Transformer

to identify the most effective model for short-term financial forecasting.

---

## 🚀 Project Overview

Financial markets are highly dynamic and influenced by complex temporal patterns, volatility, and market momentum.

This project builds an end-to-end prediction pipeline that:

✅ Collects historical and intraday stock data

✅ Engineers technical indicators

✅ Trpares rolling time-series windows

✅ Trains multiple forecasting models

✅ Generates next-day predictions

✅ Produces buy / sell / hold signals

The system focuses on **T+1 forecasting** using only quantitative market features.

---

## 🎯 Objectives

- Predict next-day stock movements
- Compare ML and DL forecasting models
- Evaluate forecasting accuracy
- Support real-time decision making
- Build reusable prediction workflows

---

## 📊 Dataset

Data sources:

- Yahoo Finance (yfinance)
- Market index data

Assets used:

Technology stocks including:

- AAPL
- MSFT
- GOOGL
- AMZN
- NVDA
- META

Market benchmarks:

- S&P 500
- NASDAQ

Data types:

- OHLCV
- Daily prices
- Intraday prices
- Technical indicators

---

## ⚙ Feature Engineering

Raw market data was transformed into predictive features:

### Price Features

- Daily Returns
- Lagged Returns
- High-Low Spread

### Volatility Features

- Rolling Volatility
- Price Range Metrics

### Momentum Features

- Moving Average (5-day)
- Moving Average (10-day)
- Relative Strength Index (RSI)

### Relative Features

- Relative Strength Ratios
- Market Index Relationships

Rolling windows:

```text
Window Size = 7 Days
Prediction Horizon = T+1
```

---

## 🧠 Models Implemented

### Traditional ML

### Random Forest

Strengths:

- Nonlinear relationships
- Fast training
- Strong baseline performance

---

### XGBoost

Strengths:

- Boosted ensembles
- Feature importance
- Noise handling

---

### Time Series

### ARIMA

Used as classical forecasting baseline.

Captures:

- Trend
- Seasonality
- Autoregressive behaviour

---

### Deep Learning

### LSTM

Architecture:

```text
Input Sequence
      ↓
LSTM Layer
      ↓
LSTM Layer
      ↓
Dense Layer
      ↓
Prediction
```

Training techniques:

- Early stopping
- ReduceLROnPlateau
- Smooth L1 Loss
- Adam optimizer

---

### Transformer

Transformer-based forecasting achieved strongest performance.

Advantages:

- Self-attention
- Long-range dependency capture
- Parallel processing
- Better temporal modelling

Performance:

```text
RMSE < 1%
```

---

## 🏗 System Architecture

```text
Market Data Collection
          ↓
Preprocessing
          ↓
Feature Engineering
          ↓
Rolling Window Generation
          ↓
Model Training
(RF / XGB / ARIMA / LSTM / Transformer)
          ↓
Prediction Engine
          ↓
Buy / Sell / Hold Signals
```

---

## 🛠 Tech Stack

Languages:

- Python

Libraries:

- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib

Models:

- Random Forest
- XGBoost
- ARIMA
- LSTM
- Transformer

Utilities:

- Jupyter Notebook
- Git
- CSV Pipeline

---

## 📁 Project Structure

```text
stock_prediction/

│── data/
│   ├── raw/
│   ├── processed/
│
│── notebooks/
│
│── feature_engg.py
│
│── data_cleaning.py
│
│── model_building_DL.py
│
│── lstm_prediction.py
│
│── transformer_training.py
│
│── evaluation.py
│
│── models/
│
│── outputs/
│
└── README.md
```

---

## 📈 Results

| Model | RMSE | Performance |
|----------|---------|-------------|
| Random Forest | ~1.5% | Moderate |
| XGBoost | ~1.7% | Moderate |
| ARIMA | ~1.5% | Stable |
| LSTM | ~1.2% | Strong |
| Transformer | <1.0% | Best |

Transformer achieved:

✅ Lowest prediction error

✅ Highest directional accuracy

✅ Better temporal dependency capture

---

## 🔬 Evaluation Metrics

Performance measured using:

### RMSE

Root Mean Squared Error

Measures forecast deviation.

### Directional Accuracy

Measures prediction correctness:

```text
Up / Down Movement Accuracy
```

---

## 📌 Key Outcomes

- Built end-to-end forecasting workflow
- Compared ML vs DL approaches
- Achieved <1% RMSE using Transformers
- Generated trading support signals
- Implemented reusable prediction pipeline

---

## 🔮 Future Improvements

Planned enhancements:

- Sentiment analysis integration
- Financial news processing
- Social media signals
- Real-time APIs
- Airflow orchestration
- Docker deployment
- Cloud deployment
- Portfolio optimisation
- Streamlit dashboard
- FastAPI serving layer

---

## 👥 Team

**Abhishek Teotia**

- Model development
- Forecasting workflows
- Feature engineering
- Evaluation

Other contributors:

Muskan Choudhary  
Justin Govanxa  
Pai Ye  
Jintian Liu  
Yuzhong Wang

---

## 📜 License

Educational and research use only.

Master of Data Science and Innovation  
University of Technology Sydney
