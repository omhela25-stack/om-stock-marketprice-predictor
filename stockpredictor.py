# stock_predictor_app.py
# stock_predictor_app.py
# stock_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

# ---------------- FETCH DATA ----------------
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        yf_period_map = {
            "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
            "1y": "1y", "2y": "2y", "5y": "5y"
        }
        df = yf.download(ticker, period=yf_period_map[period], interval="1d")
        if df.empty:
            raise Exception("No data from Yahoo Finance")

        df.reset_index(inplace=True)
        df.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume"
        }, inplace=True)
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

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    return model, scaler, metrics

def predict_next(model, scaler, df):
    X, _ = prepare_features(df)
    last_row = X.iloc[-1:].values
    return model.predict(scaler.transform(last_row))[0]

# ---------------- STREAMLIT UI ----------------
st.title("📈 Stock Price Predictor")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter stock ticker", "AAPL", key="ticker_input")
period = st.sidebar.selectbox(
    "Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="period_select"
)
predict_btn = st.sidebar.button("🚀 Predict", key="predict_btn")

if predict_btn:
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(ticker, period)

    if df.empty:
        st.error("No data available. Try another ticker or later.")
    else:
        df = process_data(df)

        # ✅ SAFETY CHECK: ensure enough rows for indicators
        if df.empty or len(df) < 60:
            st.error(
                f"Not enough data to calculate features for {ticker}. "
                "Try selecting a longer period (6mo or 1y)."
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
                model, scaler, metrics = train_model(df)

            st.subheader("🤖 Model Performance")
            st.write(metrics)

            # Prediction
            pred_price = predict_next(model, scaler, df)
            current_price = df["Close"].iloc[-1]
            change = pred_price - current_price
            pct = change / current_price * 100

            st.subheader("🔮 Next Day Prediction")
            st.metric("Predicted Price", f"${pred_price:.2f}", f"{pct:.2f}%")

            # Charts
            st.subheader("📈 Price Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["Date"], df["Close"], label="Close Price", color="blue")
            ax.plot(df["Date"], df["MA20"], label="MA20", linestyle="--", color="orange")
            ax.plot(df["Date"], df["MA50"], label="MA50", linestyle=":", color="green")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            st.pyplot(fig)

            st.subheader("📊 Volume Chart")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(df["Date"], df["Volume"], color="skyblue")
            st.pyplot(fig)

            st.subheader("📉 RSI Chart")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df["Date"], df["RSI"], color="red")
            ax.axhline(70, linestyle="--", color="orange")
            ax.axhline(30, linestyle="--", color="green")
            ax.set_ylim(0, 100)
            st.pyplot(fig)

            # Data Table
            st.subheader("📋 Recent Data")
            st.dataframe(df.tail(20))
























