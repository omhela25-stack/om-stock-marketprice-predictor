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
ALPHA_VANTAGE_API_KEY = "LPRQX827JWWLKA4R"   # replace with your key
AV_BASE_URL = "https://www.alphavantage.co/query"

st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

# ---------------- FETCH DATA ----------------
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="1Y"):
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
        days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825}[period]
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

def process_data(df, ma1=20, ma2=50):
    """Add technical indicators + lags"""
    df[f"MA{ma1}"] = df["Close"].rolling(ma1).mean()
    df[f"MA{ma2}"] = df["Close"].rolling(ma2).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df["Return"] = df["Close"].pct_change()
    for lag in [1, 2, 3, 5]:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    return df.dropna()

def prepare_features(df, ma1=20, ma2=50):
    features = ["Open", "High", "Low", "Volume", f"MA{ma1}", f"MA{ma2}", 
                "RSI", "Return", "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5"]
    X = df[features].fillna(0)
    y = df["Close"]
    return X, y

# ---------------- MODEL ----------------
def train_model(df, ma1=20, ma2=50):
    X, y = prepare_features(df, ma1, ma2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    return model, scaler, metrics

def predict_next(model, scaler, df, ma1=20, ma2=50):
    X, _ = prepare_features(df, ma1, ma2)
    last_row = X.iloc[-1:].values
    return model.predict(scaler.transform(last_row))[0]

# ---------------- STREAMLIT UI ----------------
st.title("📈 Stock Price Predictor")

# Sidebar
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.text_input("Enter stock ticker", "AAPL", key="ticker_input")
period = st.sidebar.selectbox("Period", ["1M", "3M", "6M", "1Y", "2Y", "5Y"], index=3, key="period_select")

# Moving averages
ma1 = st.sidebar.number_input("Short MA", value=20, min_value=5, max_value=50, step=1, key="ma1")
ma2 = st.sidebar.number_input("Long MA", value=50, min_value=10, max_value=200, step=5, key="ma2")

# RSI thresholds
rsi_upper = st.sidebar.slider("RSI Overbought", 60, 90, 70, key="rsi_upper")
rsi_lower = st.sidebar.slider("RSI Oversold", 10, 40, 30, key="rsi_lower")

predict_btn = st.sidebar.button("🚀 Predict", key="predict_btn")

if predict_btn:
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(ticker, period)

    if df.empty:
        st.error("No data available. Try another ticker or later.")
    else:
        df = process_data(df, ma1, ma2)

        # ✅ SAFETY CHECK
        if df.empty or len(df) < max(ma1, ma2) + 10:
            st.error(
                f"Not enough data to calculate features for {ticker}. "
                "Try selecting a longer period."
            )
        else:
            # Current stats
            st.subheader(f"📊 {ticker} - Latest Data")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            col3.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

            # Train
            with st.spinner("Training model..."):
                model, scaler, metrics = train_model(df, ma1, ma2)

            st.subheader("🤖 Model Performance")
            st.write(metrics)

            # Prediction
            pred_price = predict_next(model, scaler, df, ma1, ma2)
            current_price = df["Close"].iloc[-1]
            change = pred_price - current_price
            pct = change / current_price * 100

            st.subheader("🔮 Next Day Prediction")
            st.metric("Predicted Price", f"${pred_price:.2f}", f"{pct:.2f}%")

            # Charts
            st.subheader("📈 Price Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["Date"], df["Close"], label="Close Price", color="blue")
            ax.plot(df["Date"], df[f"MA{ma1}"], label=f"MA{ma1}", linestyle="--", color="orange")
            ax.plot(df["Date"], df[f"MA{ma2}"], label=f"MA{ma2}", linestyle=":", color="green")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            st.pyplot(fig)

            st.subheader("📉 RSI Chart")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df["Date"], df["RSI"], color="red")
            ax.axhline(rsi_upper, linestyle="--", color="orange")
            ax.axhline(rsi_lower, linestyle="--", color="green")
            ax.set_ylim(0, 100)
            st.pyplot(fig)

            st.subheader("📊 Volume Chart")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(df["Date"], df["Volume"], color="skyblue")
            st.pyplot(fig)

            # Data Table
            st.subheader("📋 Recent Data")
            st.dataframe(df.tail(20))


























