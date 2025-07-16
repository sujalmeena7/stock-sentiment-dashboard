# Real-Time Stock Sentiment & Forecast Dashboard

An interactive Streamlit app that combines **real-time stock price visualization**, **news-based sentiment analysis**, and a **7-day stock price forecast** using an LSTM neural network. Perfect for finance enthusiasts, data scientists, and investors.

---

##  Features

###  Stock Price Visualization
- Interactive **Plotly chart** for any stock ticker (e.g., `AAPL`, `TSLA`, `INFY`)
- Last 6 months of historical data via Yahoo Finance
- Hover-enabled tooltips and zooming

###  News Sentiment Analysis
- Live news headlines from **Google News**
- Sentiment scoring using **VADER**
- Color-coded (green/red) sentiment table
- Refresh button to fetch the latest headlines
- Overall average sentiment score displayed

###  LSTM-Based Price Forecast
- Predicts **next 7 days of stock prices**
- Trained on last 6 months of closing prices
- Shows RMSE (model accuracy)
- Forecast line color: **green** (bullish) or **red** (bearish)
- % change from last actual to first forecasted price
- Hoverable, dashed forecast line
- Optional debug info toggle

###  Model Info
Expandable section explaining the model, assumptions, and performance.

---



##  Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | Frontend app |
| `yfinance` | Real-time stock data |
| `Google News RSS` | Headline scraping |
| `VADER` | Sentiment scoring |
| `TensorFlow/Keras` | LSTM model |
| `scikit-learn` | Data scaling (MinMaxScaler) |
| `Plotly` | Visualizations |

---

##  Folder Structure

stock-sentiment-dashboard/
│
├── app/
│ └── main.py # Streamlit frontend
│
├── models/
│ └── lstm_model.py # LSTM model logic
│
├── utils/
│ ├── stock_data.py # Yahoo Finance fetching
│ └── news_sentiment.py # News scraping + sentiment
│
├── data/ #  For saved runs
├── requirements.txt # Dependencies
└── README.md # You're here!

