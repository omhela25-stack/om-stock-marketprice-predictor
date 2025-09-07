# stock_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
ALPHA_VANTAGE_API_KEY = "LPRQX827JWWLKA4R"   # ⚠️ replace with your own key (better: use secrets)
AV_BASE_URL = "https://www.alphavantage.co/query"

st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

# ---------------- FETCH DATA ----------------
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Alpha Vantage"""
    try:
        time.sleep(1)  # avoid hitting rate limit
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "full",
            "datatype": "json"
        }
        response = requests.get(AV_BASE_URL, params=params, timeout=15)
        data = response.json()

        if "Time Series (Daily)" not in data:
            raise Exception("API returned no data")

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        })
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={"index": "Date"})

        # filter by period
        days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}[period]
        df = df[df["Date"] >= datetime.now() - timedelta(days=days)]
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return pd.DataFrame()

# ---------------- FEATURES ----------------
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_data(df):
    """Add technical indicators + lags"""
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df["Return"] = df["Close"].pct_change()
    for lag in [1, 2, 3, 5]:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    return df.dropna()

def prepare_features(df):
    features = ["Open", "High", "Low", "Volume", "MA20", "MA50", "RSI", "Return",
                "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5"]
    X = df[features].fillna(0)
    y = df["Close"]
    return X, y

# ---------------- MODEL ----------------
def train_model(df):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
# stock_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
ALPHA_VANTAGE_API_KEY = "LPRQX827JWWLKA4R"   # ⚠️ replace with your own key (better: use secrets)
AV_BASE_URL = "https://www.alphavantage.co/query"

st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

# ---------------- FETCH DATA ----------------
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Alpha Vantage"""
    try:
        time.sleep(1)  # avoid hitting rate limit
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "full",
            "datatype": "json"
        }
        response = requests.get(AV_BASE_URL, params=params, timeout=15)
        data = response.json()

        if "Time Series (Daily)" not in data:
            raise Exception("API returned no data")

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        })
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={"index": "Date"})

        # filter by period
        days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}[period]
        df = df[df["Date"] >= datetime.now() - timedelta(days=days)]
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return pd.DataFrame()

# ---------------- FEATURES ----------------
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_data(df):
    """Add technical indicators + lags"""
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df["Return"] = df["Close"].pct_change()
    for lag in [1, 2, 3, 5]:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    return df.dropna()

def prepare_features(df):
    features = ["Open", "High", "Low", "Volume", "MA20", "MA50", "RSI", "Return",
                "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5"]
    X = df[features].fillna(0)
    y = df["Close"]
    return X, y

# ---------------- MODEL ----------------
def train_model(df):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)














