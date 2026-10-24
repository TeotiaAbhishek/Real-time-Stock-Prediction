# Stock Price Forecasting: Transformer vs LSTM vs ARIMA

Can deep learning beat classical time series models at predicting tomorrow's stock price? This project builds an end-to-end pipeline that pulls two years of daily data for the 10 largest technology stocks from Yahoo Finance, engineers market and momentum features, trains a Transformer, an LSTM and an ARIMA model per stock, and compares their forecasts against what the market actually did.

Built as a team project for the Master of Data Science and Innovation at the University of Technology Sydney.

![NVDA predicted vs actual close prices](charts/NVDA.png)

## Results

Forecasts were made for 29 April 2025 (next day) and the following four trading days, then compared with actual closing prices for all 10 stocks.

| Model | Next-day error (mean absolute %) | Best model on | 5-day error (mean absolute %) |
|---|---:|---:|---:|
| Transformer | 1.04% | 5 of 10 stocks | 3.46% |
| ARIMA (5,0,2) | 1.05% | 2 of 10 stocks | 3.15% |
| LSTM | 1.07% | 3 of 10 stocks | 3.20% |
| Naive baseline (yesterday's close) | 0.75% | | |

**What this shows**

- The three models are within 0.03 percentage points of each other on next-day error. The Transformer was most often the best per stock, but no model clearly wins.
- None of them beat the naive baseline of assuming tomorrow's close equals today's. This is a common and honest result for daily price forecasting: most day-to-day movement is noise that price and volume features can't explain.
- Error roughly triples over five days, and all models miss sharp moves, such as NVDA's 5% rise on 1–2 May 2025 (chart above).
- The evaluation covers one week, so treat these numbers as indicative. A rolling backtest over many months is the next step (see [Next steps](#next-steps)).

Per-stock results are in [`next_day_model_metrics.csv`](next_day_model_metrics.csv), and charts for every stock are in [`charts/`](charts/) (interactive versions in [`charts/charts_plotly/`](charts/charts_plotly/)).

## Pipeline

```text
Yahoo Finance (10 tech stocks + S&P 500 + NASDAQ, 2 years daily)
        |
Data/data_prep.py        download prices, analyst upgrades/downgrades and indices
Data/data_cleaning.py    drop empty target rows, fill missing analyst ratings
        |
Features/feature_engg.py engineer ~25 features per stock
        |
Model/model_building_DL.py   Transformer  (PyTorch)
Model/lstm.py                LSTM         (PyTorch)
Model/arima.py               ARIMA(5,0,2) (statsmodels)
        |
Model/predict_*.py       5-day forecasts per model
test.py                  compare with actual closes, write metrics and Plotly charts
```

`main.py` runs every step in order.

## Data

- **Stocks:** the 10 largest companies in Yahoo Finance's technology sector at run time. In this run: AAPL, AVGO, CRM, CSCO, IBM, MSFT, NOW, NVDA, ORCL and PLTR.
- **Market context:** S&P 500 and NASDAQ Composite closes.
- **Analyst ratings:** upgrades and downgrades per stock from Yahoo Finance.
- **Window:** two years of daily OHLCV data, ending the day before the run.

## Features

For each stock the models predict the next day's return, which is converted back to a closing price.

| Group | Features |
|---|---|
| Price | daily return, high-low range, intraday volatility |
| Momentum | 5-day and 10-day returns, 14-day RSI |
| Volatility | 5-day rolling standard deviation of returns |
| Lags | returns from the previous 1 to 5 days |
| Market | S&P 500 and NASDAQ daily returns, stock-to-index price ratios |
| Analyst | flag for a rating change that day |
| Calendar | day of week |

## Models

| Model | Setup |
|---|---|
| Transformer | Linear projection to 128 dimensions, 2 encoder layers with 4 attention heads, dropout 0.2, 7-day input window |
| LSTM | 4 layers, 256 hidden units, 7-day input window |
| ARIMA | order (5,0,2) on daily returns |

Both neural networks use Smooth L1 loss, the Adam optimiser (learning rate 0.001), learning-rate reduction on plateau, early stopping (patience 20) and a time-ordered 80/20 train/validation split, so the model never sees future data during training. Features and targets are standardised per stock, and the fitted scalers are saved with each model.

## How to run

```bash
pip install -r requirements.txt
python main.py     # download data, build features, train all models, forecast
python test.py     # compare forecasts with actual closes, write metrics and interactive charts
```

Trained models for the run above are saved in `models/`, so you can skip training and run the prediction scripts in `Model/` directly.

Note that `main.py` downloads fresh data, so a new run uses a different date range and may pick different stocks.

## Project structure

```text
├── main.py                      runs the full pipeline
├── test.py                      evaluation, metrics and charts
├── Data/                        download and cleaning
├── Features/feature_engg.py     feature engineering
├── Model/                       training and prediction for each model
├── models/                      saved models, scalers and feature lists
├── charts/                      predicted vs actual charts per stock
├── next_day_model_metrics.csv   next-day results per stock and model
└── *_predicted_closes_5days.csv 5-day forecasts per model
```

## Next steps

- Rolling-origin backtest over 6 to 12 months instead of one week, with directional accuracy alongside error.
- Always report a naive baseline, and test whether models add value on returns rather than price level.
- Add news and sentiment features, which carry information that price history doesn't.
- Serve forecasts through a small API or Streamlit dashboard and schedule daily retraining.

## Team

Abhishek Teotia (model development, feature engineering, forecasting workflow and evaluation), with Muskan Choudhary, Justin Govanxa, Pai Ye, Jintian Liu and Yuzhong Wang.

Built for the Master of Data Science and Innovation, University of Technology Sydney. For educational use only, not investment advice.
