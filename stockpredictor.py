import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# -------------- CONFIG & STYLE --------------
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
def local_css(css_text: str):
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
custom_css = """
/* App background */
.main {background-color: #E0115F ; color: #FF1493;font-family: 'Trebuchet MS', sans-serif;}
[data-testid="stSidebar"] {background-color: #1E1E2F; color: white;}
h1, h2, h3 {color: #009999 !important; font-weight: bold;}
[data-testid="stMetricValue"] {color: #39FF14 !important;font-size: 28px;}
[data-testid="stMetricDelta"] {color: #FFD700 !important;font-size: 18px;}
.stButton>button {background: linear-gradient(90deg, #FF4B4B, #FF9900); color: white;
border-radius: 8px;font-weight: bold;}
"""
plt.style.use("seaborn-v0_8-darkgrid")
local_css(custom_css)

import streamlit as st
import yfinance as yf
import time
import pandas as pd

# Function to fetch live prices for a list of tickers
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_latest_prices(tickers):
    prices = {}
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                prices[ticker] = data["Close"].iloc[-1]
            else:
                prices[ticker] = None
        except Exception:
            prices[ticker] = None
    return prices

# Your tickers list (can replace with dynamic from your app)
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "RELIANCE.NS", "TCS.NS"]

# Fetch prices
prices = get_latest_prices(tickers)

# Build marquee text with ticker and price
marquee_text = "  ⚫  ".join(
    [f"{t} (${prices[t]:.2f})" if prices[t] is not None else f"{t} (N/A)" for t in tickers]
)

# Marquee HTML + CSS for sliding effect
marquee_html = f"""
<div style="white-space: nowrap; overflow: hidden; width: 100%; background-color:#1E1E2F; color:#39FF14; font-weight:bold; padding: 5px 0;">
  <div style="
    display: inline-block;
    padding-left: 100%;
    animation: marquee 25s linear infinite;
    font-family: 'Trebuchet MS', sans-serif;
    font-size: 16px;
  ">
    {marquee_text}
  </div>
</div>
<style>
@keyframes marquee {{
  0%   {{ transform: translateX(0%); }}
  100% {{ transform: translateX(-100%); }}
}}
</style>
"""

# Display at top of the app
st.markdown(marquee_html, unsafe_allow_html=True)

# --- Rest of your app code below ---

# Example: Use the ticker from dropdown or custom input as you do in your app
# ... Your existing sidebar and main app logic ...

# -------------- TICKER + COMPANY NAME LOGIC ---------------
@st.cache_data(show_spinner=True, ttl=6*3600)
def load_full_tickers(include_global=True, top_n=500):
    tickers = []
    names = []
    try:
        st.info("⏳ Fetching S&P500 / NASDAQ tickers & company names (first run can take ~20 sec)...")
        sp500 = yf.tickers_sp500()
        nasdaq = yf.tickers_nasdaq()
        base = list(set(sp500 + nasdaq))[:top_n]
        data = []
        for t in base:
            try:
                info = yf.Ticker(t).info
                name = info.get('shortName') or info.get('longName') or t
                data.append({"Symbol": t, "Name": name})
            except Exception:
                data.append({"Symbol": t, "Name": t})
            time.sleep(0.04)  # be nice to Yahoo! API
        tickers_df = pd.DataFrame(data)
        tickers = tickers_df["Symbol"].fillna("").tolist()
        names = tickers_df["Name"].fillna("").tolist()
    except Exception:
        tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NFLX", "RELIANCE.NS", "TCS.NS"]
        names = ["Apple", "Microsoft", "Tesla", "Google", "Amazon", 
                 "Netflix", "Reliance Industries", "Tata Consultancy"]
    if include_global:
        extra = [
            ("RELIANCE.NS", "Reliance Industries"),
            ("TCS.NS", "Tata Consultancy"),
            ("INFY.NS", "Infosys"),
            ("HSBA.L", "HSBC Holdings"),
            ("BMW.DE", "BMW AG"),
            ("DAI.DE", "Mercedes-Benz"),
        ]
        for symbol, name in extra:
            if symbol not in tickers:
                tickers.append(symbol)
                names.append(name)
    return tickers, names

def format_options(tickers, names):
    return {f"{sym} - {nam}": sym for sym, nam in zip(tickers, names)}

# -------------- FETCH DATA -----------------
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="1Y"):
    try:
        period_map = {
            "1M": "1mo", "3M": "3mo", "6M": "6mo",
            "1Y": "1y", "2Y": "2y", "5Y": "5y"
        }
        yf_period = period_map.get(period, "1y")
        df = yf.download(ticker, period=yf_period, interval="1d", progress=False)
        if df.empty:
            raise Exception("No data returned from Yahoo Finance")
        df = df.reset_index()
        return df.rename(columns={
            "Date": "Date", "Open": "Open", "High": "High",
            "Low": "Low", "Close": "Close", "Volume": "Volume"
        })
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return pd.DataFrame()

# ------------- FEATURE ENGINEERING ----------
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_data(df, ma1=20, ma2=50):
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

# -------------- MODEL ------------------
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
    pred = model.predict(scaler.transform(last_row))
    if isinstance(pred, (pd.Series, np.ndarray)):
        return pred[0]
    return pred

# ---------------- STREAMLIT UI ----------------
st.title("📈 Stock Price Predictor (Global + Custom)")

st.sidebar.header("⚙️ Settings")

tickers, names = load_full_tickers()
ticker_mode = st.sidebar.radio(
    "Ticker input mode:",
    ("🔍 Search by name", "🔤 Enter custom")
)

if ticker_mode == "🔍 Search by name":
    options = format_options(tickers, names)
    ticker_display = st.sidebar.selectbox("Search or pick stock", list(options.keys()))
    ticker = options[ticker_display]
else:
    ticker = st.sidebar.text_input("Enter custom stock ticker (e.g., RELIANCE.NS)", "AAPL")

period = st.sidebar.selectbox("Period", ["1M", "3M", "6M", "1Y", "2Y", "5Y"], index=3)
ma1 = st.sidebar.number_input("Short MA", value=20, min_value=5, max_value=50, step=1)
ma2 = st.sidebar.number_input("Long MA", value=50, min_value=10, max_value=200, step=5)
rsi_upper = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_lower = st.sidebar.slider("RSI Oversold", 10, 40, 30)
predict_btn = st.sidebar.button("🚀 Predict")

if predict_btn:
    st.markdown(f"## 📊 Results for **{ticker}**")
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_stock_data(ticker, period)
    if df.empty:
        st.error(f"No data available for {ticker}.")
    else:
        df = process_data(df, ma1, ma2)
        if df.empty or len(df) < max(ma1, ma2) + 10:
            st.error(f"Not enough data to calculate features for {ticker}.")
        else:
            # Display current stats
            col1, col2, col3 = st.columns(3)
            try:
                last_close = df["Close"].iloc[-1]
                last_volume = df["Volume"].iloc[-1]
                last_rsi = df["RSI"].iloc[-1]
                # Ensure scalar floats
                if isinstance(last_close, (pd.Series, np.ndarray)):
                    last_close = float(last_close.item())
                if isinstance(last_volume, (pd.Series, np.ndarray)):
                    last_volume = float(last_volume.item())
                if isinstance(last_rsi, (pd.Series, np.ndarray)):
                    last_rsi = float(last_rsi.item())
                col1.metric("Current Price", f"${last_close:.2f}")
                col2.metric("Volume", f"{last_volume:,.0f}")
                col3.metric("RSI", f"{last_rsi:.2f}")
            except Exception as e:
                st.warning(f"⚠️ Could not display latest stats: {e}")

            # Train model
            with st.spinner(f"Training model for {ticker}..."):
                model, scaler, metrics = train_model(df, ma1, ma2)
            st.subheader("🤖 Model Performance")
            st.write(metrics)

            # Prediction with scalar fix
            pred_price = predict_next(model, scaler, df, ma1, ma2)
            if isinstance(pred_price, (pd.Series, np.ndarray)):
                pred_price = pred_price.item() if hasattr(pred_price, 'item') else pred_price[0]

            current_price = df["Close"].iloc[-1]
            change = pred_price - current_price
            pct = (change / current_price) * 100
            if isinstance(pct, (pd.Series, np.ndarray)):
                pct = pct.item() if hasattr(pct, 'item') else pct[0]

            st.metric("🔮 Predicted Price", f"${pred_price:.2f}", f"{pct:.2f}%")

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
            try:
                fig, ax = plt.subplots(figsize=(10, 3))
                dates = pd.to_datetime(df["Date"]).dt.to_pydatetime().tolist()
                volumes = df["Volume"].squeeze().astype(float).tolist()
                if len(dates) != len(volumes):
                    raise ValueError(f"Length mismatch: {len(dates)} dates vs {len(volumes)} volumes")
                ax.bar(dates, volumes, color="skyblue")
                ax.set_xlabel("Date")
                ax.set_ylabel("Volume")
                fig.autofmt_xdate()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Could not render Volume chart: {e}")

            st.subheader("📋 Recent Data")
            st.dataframe(df.tail(20))

# Footer
st.markdown(
    """
    <div style="text-align: center; color: pink ; padding: 10px; font-size: 14px;">
        Made by <b> Om Hela , </b>(Minor in AI) IIT ROPAR
    </div>
    """,
    unsafe_allow_html=True
)















































