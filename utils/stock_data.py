# utils/stock_data.py

import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock price data using yfinance.
    
    :param ticker: Stock symbol (e.g., "AAPL", "TSLA")
    :param period: Data period (e.g., "1mo", "6mo", "1y", "5y")
    :param interval: Data interval (e.g., "1d", "1h", "1wk")
    :return: DataFrame with stock prices
    """
    data = yf.download(ticker, period=period, interval=interval)
    data.reset_index(inplace=True)
    return data
