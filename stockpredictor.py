import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ---------------- Page config & basic style ----------------
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
/* App background & text */
[data-testid="stAppViewContainer"] {
    background-color: #0f0f14;
    color: #E6E6FA;
    font-family: 'Roboto', 'Trebuchet MS', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #11111a;
    color: #ffffff;
}

/* Header */
h1, h2, h3 {
    color: #00FFFF !important;
    font-family: 'Roboto', 'Trebuchet MS', sans-serif;
    font-weight: bold;
}

/* Metrics */
[data-testid="stMetricValue"] { color: #39FF14 !important; font-size: 24px; }
[data-testid="stMetricDelta"] { color: #FFD700 !important; font-size: 14px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #FF4B4B, #FF9900);
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

/* Card like containers */
.card {
    background:#12121a;
    padding:12px;
    border-radius:10px;
    border:1px solid #222;
    margin-bottom:10px;
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)
import streamlit as st
import requests
from typing import Dict, List

st.set_page_config(page_title="Top Products Catalog", layout="wide")

# ------------------------------------------
# 🔐 Load SerpAPI Key
# ------------------------------------------
SERP_API_KEY = st.secrets["api_keys"]["serpapi"]
SERP_SEARCH_URL = "https://serpapi.com/search.json"


# ------------------------------------------
# 🔍 Function: Fetch Live Best Prices
# ------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_best_prices(product_name: str):
    """Fetch best retailer prices using SerpAPI Google Shopping."""
    params = {
        "engine": "google_shopping",
        "q": product_name,
        "api_key": SERP_API_KEY,
        "gl": "us"
    }

    try:
        res = requests.get(SERP_SEARCH_URL, params=params).json()

        items = res.get("shopping_results", [])
        if not items:
            return []

        results = []
        for item in items[:3]:  # top 3 offers
            results.append({
                "title": item.get("title", ""),
                "price": item.get("price", ""),
                "source": item.get("source", ""),
                "link": item.get("link", "")
            })
        return results

    except Exception as e:
        return []


# ---------------------------------------------------
# 📦 Curated Catalog (Top 10 per Category)
# ---------------------------------------------------
catalogs: Dict[str, List[Dict]] = {

    # ---------------- SMARTPHONES -------------------
    "Smartphones": [
        {
            "name": "Samsung Galaxy S25 Ultra",
            "review": "Top-tier camera, display & productivity features.",
            "benefits": ["Best zoom camera", "Bright OLED", "S Pen features", "Flagship power"]
        },
        {
            "name": "Google Pixel 10 Pro",
            "review": "Best computational photography & clean Android.",
            "benefits": ["AI features", "Smooth UI", "Stunning still photos"]
        },
        {
            "name": "iPhone 17 Pro Max",
            "review": "Refined cameras, performance & strong ecosystem.",
            "benefits": ["Best video", "Fast chip", "Long OS support"]
        },
        {
            "name": "OnePlus 15",
            "review": "Flagship performance + excellent fast charging.",
            "benefits": ["Long battery life", "Great value", "Fast charging"]
        },
        {
            "name": "iPhone 17",
            "review": "Best all-rounder for most users.",
            "benefits": ["Balanced performance", "Great camera"]
        },
        {
            "name": "Google Pixel 10",
            "review": "Strong camera + clean Android + AI features.",
            "benefits": ["AI tools", "Affordable flagship"]
        },
        {
            "name": "Nothing Phone 3a Pro",
            "review": "Unique design & great value.",
            "benefits": ["Glyph interface", "Clean UI"]
        },
        {
            "name": "Samsung Galaxy Z Flip 7",
            "review": "Most polished compact foldable.",
            "benefits": ["Compact", "Improved hinge"]
        },
        {
            "name": "Google Pixel 9a",
            "review": "Best budget camera phone.",
            "benefits": ["Value", "Clean software"]
        },
        {
            "name": "OnePlus 15R",
            "review": "Flagship-like features at lower price.",
            "benefits": ["Great performance", "Fast charging"]
        }
    ],

    # ---------------- LAPTOPS -------------------
    "Laptops": [
        {"name": "MacBook Pro (M-Series)", "review": "Best performance + battery life.", "benefits": ["Silent", "Efficient", "Great display"]},
        {"name": "Dell XPS 15", "review": "Premium Windows laptop.", "benefits": ["Bezel-less design", "Strong performance"]},
        {"name": "ASUS ROG Zephyrus G16", "review": "Best gaming ultrabook.", "benefits": ["Slim", "Great GPU"]},
        {"name": "HP Spectre x360", "review": "Best 2-in-1 laptop.", "benefits": ["Premium build", "Touch display"]},
        {"name": "Lenovo ThinkPad X1 Carbon", "review": "Best business laptop.", "benefits": ["Durable", "Great keyboard"]},
        {"name": "MacBook Air (M-Series)", "review": "Best lightweight laptop.", "benefits": ["Super light", "Excellent battery"]},
        {"name": "ASUS Zenbook 14 OLED", "review": "Excellent OLED screen + value.", "benefits": ["OLED", "Portable"]},
        {"name": "Acer Swift Go", "review": "Great performance/value ratio.", "benefits": ["Affordable", "Fast"]},
        {"name": "MSI Creator Z17", "review": "Best for creators.", "benefits": ["Powerful GPU", "Color-accurate"]},
        {"name": "Samsung Galaxy Book 4 Pro", "review": "Great display + thin design.", "benefits": ["Thin", "Bright AMOLED"]},
    ],

    # ---------------- HEADPHONES -------------------
    "Headphones": [
        {"name": "Sony WH-1000XM6", "review": "Best ANC over-ear.", "benefits": ["Top noise canceling", "Comfort"]},
        {"name": "Bose QuietComfort Ultra", "review": "Comfort king with natural sound.", "benefits": ["Lightweight", "Best comfort"]},
        {"name": "Apple AirPods Max 2", "review": "Best for Apple users.", "benefits": ["Spatial audio", "Build quality"]},
        {"name": "Sennheiser Momentum 5", "review": "Best sound quality.", "benefits": ["Rich sound", "Great build"]},
        {"name": "Sony WF-1000XM6", "review": "Best noise-canceling earbuds.", "benefits": ["Great ANC", "Compact"]},
        {"name": "AirPods Pro 3", "review": "Great ANC + iOS features.", "benefits": ["Spatial audio", "Comfort"]},
        {"name": "Nothing Ear 3", "review": "Best design/value earbuds.", "benefits": ["Transparent design", "Good value"]},
        {"name": "Bose QC Earbuds II", "review": "Outstanding ANC.", "benefits": ["Great isolation"]},
        {"name": "Beats Studio Pro", "review": "Great for bass lovers.", "benefits": ["Punchy sound"]},
        {"name": "JBL Tour One M2", "review": "Value ANC option.", "benefits": ["Good ANC", "Affordable"]},
    ],

    # ---------------- SMARTWATCHES -------------------
    "Smartwatches": [
        {"name": "Apple Watch Ultra 3", "review": "Best overall smartwatch.", "benefits": ["Rugged", "Best sensors"]},
        {"name": "Apple Watch Series 10", "review": "Best mainstream option.", "benefits": ["Lightweight", "Accurate tracking"]},
        {"name": "Galaxy Watch 7 Pro", "review": "Best Android smartwatch.", "benefits": ["Long battery", "Great display"]},
        {"name": "Google Pixel Watch 3", "review": "Best Google AI watch.", "benefits": ["AI features", "Clean design"]},
        {"name": "Amazfit GTR 5", "review": "Best budget smartwatch.", "benefits": ["Long battery", "Good fitness"]},
        {"name": "Garmin Fenix 8", "review": "Best for athletes.", "benefits": ["Advanced metrics", "Rugged"]},
        {"name": "Garmin Venu 3", "review": "Great fitness features.", "benefits": ["Training tools"]},
        {"name": "Fitbit Versa 5", "review": "Affordable fitness tracker.", "benefits": ["Good tracking"]},
        {"name": "Huawei Watch GT5", "review": "Long battery.", "benefits": ["Battery", "Premium build"]},
        {"name": "Nothing Watch Pro", "review": "Stylish & affordable.", "benefits": ["Clean UI"]},
    ],

    # ---------------- CAMERAS -------------------
    "Cameras": [
        {"name": "Sony A7 IV", "review": "Best hybrid mirrorless.", "benefits": ["Great AF", "Image quality"]},
        {"name": "Canon R6 Mark II", "review": "Excellent hybrid system.", "benefits": ["Fast AF", "Great video"]},
        {"name": "Nikon Z6 III", "review": "Fantastic for hybrid shooting.", "benefits": ["Dynamic range"]},
        {"name": "Sony A6700", "review": "Best APS-C camera.", "benefits": ["AF", "Compact"]},
        {"name": "Fujifilm X-T5", "review": "Top APS-C for creators.", "benefits": ["Great colors"]},
        {"name": "Panasonic GH6", "review": "Best for video creators.", "benefits": ["Video tools"]},
        {"name": "Sony A1", "review": "Flagship all-rounder.", "benefits": ["8K", "Fast sensor"]},
        {"name": "Canon R5", "review": "Great pro hybrid.", "benefits": ["High res", "AF"]},
        {"name": "Sony ZV-E10 II", "review": "Best vlogging camera.", "benefits": ["Flip screen"]},
        {"name": "Fujifilm GFX100 II", "review": "Best medium format.", "benefits": ["Insane detail"]},
    ],

    # ---------------- TABLETS -------------------
    "Tablets": [
        {"name": "iPad Pro M4", "review": "Best tablet overall.", "benefits": ["OLED", "Strongest chip"]},
        {"name": "iPad Air M3", "review": "Best value iPad.", "benefits": ["Fast", "Affordable"]},
        {"name": "Samsung Tab S10 Ultra", "review": "Best Android tablet.", "benefits": ["Huge AMOLED"]},
        {"name": "Samsung Tab S10+", "review": "Great premium Android.", "benefits": ["OLED", "Stylus"]},
        {"name": "Xiaomi Pad 7 Pro", "review": "Best midrange Android.", "benefits": ["Affordable"]},
        {"name": "Lenovo Tab P13 Pro", "review": "Great for media.", "benefits": ["Great display"]},
        {"name": "Amazon Fire Max 11", "review": "Budget media tablet.", "benefits": ["Cheap"]},
        {"name": "iPad Mini 7", "review": "Portable powerhouse.", "benefits": ["Compact"]},
        {"name": "Huawei MatePad Pro", "review": "Premium build.", "benefits": ["Stylus support"]},
        {"name": "Realme Pad 2", "review": "Best budget tablet.", "benefits": ["Affordable"]},
    ],

    # ---------------- TVs -------------------
    "TVs": [
        {"name": "LG G4 OLED", "review": "Best overall OLED.", "benefits": ["Brightness", "Colors"]},
        {"name": "Samsung S95D OLED", "review": "Best QD-OLED.", "benefits": ["Contrast", "Brightness"]},
        {"name": "Sony A95L", "review": "Excellent picture quality.", "benefits": ["Colors", "Processing"]},
        {"name": "TCL QM8", "review": "Best value Mini-LED.", "benefits": ["Brightness", "Price"]},
        {"name": "Hisense U8K", "review": "Great value high-end TV.", "benefits": ["Mini-LED"]},
        {"name": "Samsung QN90D", "review": "Premium Mini-LED option.", "benefits": ["Brightness"]},
        {"name": "LG C4", "review": "Great OLED value.", "benefits": ["Contrast"]},
        {"name": "Sony X90L", "review": "Strong mid-range TV.", "benefits": ["Motion handling"]},
        {"name": "TCL Q7", "review": "Affordable 4K option.", "benefits": ["Value"]},
        {"name": "Hisense U7N", "review": "Good gaming TV.", "benefits": ["Low latency"]},
    ]
}


# ---------------------------------------------------
# 🎨 UI Rendering
# ---------------------------------------------------
st.title("🔥 Top 10 Best Products — Live Prices + Reviews")
st.markdown("Browse curated top picks with **live price comparison** from multiple retailers.")

category = st.sidebar.selectbox("Select Category", list(catalogs.keys()))

st.subheader(f"Top 10 – {category}")

for idx, product in enumerate(catalogs[category], 1):
    st.markdown(f"### **{idx}. {product['name']}**")
    st.write(product["review"])
    st.write("**Key Benefits:** " + ", ".join(product["benefits"]))

    with st.expander("💰 Live Best Prices (via SerpAPI)"):
        with st.spinner("Fetching best prices…"):
            prices = fetch_best_prices(product["name"])

        if prices:
            for p in prices:
                st.markdown(
                    f"- **{p['source']}** — {p['price']} "
                    f"[Buy]({p['link']})"
                )
        else:
            st.info("No live prices found.")

    st.markdown("---")

# ---------------- Utilities & Caching ----------------
@st.cache_data(ttl=120, show_spinner=False)
def get_batch_prices(ticker_list):
    """Batch fetch today's close price via yf.download to reduce calls."""
    try:
        # yf.download returns columns like ('Close','AAPL') if multiple tickers.
        data = yf.download(ticker_list, period="1d", interval="1d", progress=False, threads=True)
        prices = {}
        if data.empty:
            for t in ticker_list:
                prices[t] = None
            return prices

        # If single ticker, align shape
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Close'].iloc[-1]
            for t in ticker_list:
                prices[t] = float(closes.get(t, np.nan)) if pd.notna(closes.get(t, np.nan)) else None
        else:
            # only one ticker requested
            prices[ticker_list[0]] = float(data['Close'].iloc[-1]) if not data.empty else None
            for t in ticker_list[1:]:
                prices[t] = None
        return prices
    except Exception:
        return {t: None for t in ticker_list}

@st.cache_data(ttl=6*3600, show_spinner=True)
def load_ticker_list(top_n=500, include_global=True):
    """Load a list of tickers (S&P / NASDAQ) - limited first-run overhead.
       We don't call .info for each ticker here to keep it fast."""
    # Fallback short list in case yf.tickers_* are unavailable
    try:
        sp = yf.tickers_sp500()
        nas = yf.tickers_nasdaq()
        base = list(dict.fromkeys((sp or []) + (nas or [])))[:top_n]
    except Exception:
        base = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NFLX", "RELIANCE.NS", "TCS.NS"]
    tickers = base.copy()
    names = [t for t in tickers]  # placeholder names (we'll fetch the selected ticker's name when needed)
    if include_global:
        extras = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HSBA.L", "BMW.DE", "DAI.DE"]
        for e in extras:
            if e not in tickers:
                tickers.append(e)
                names.append(e)
    return tickers, names

# ---------------- Data fetching ----------------
@st.cache_data(ttl=300, show_spinner=True)
def fetch_stock_history(ticker, period="1Y"):
    period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}
    yf_period = period_map.get(period, "1y")
    try:
        df = yf.download(ticker, period=yf_period, interval="1d", progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"Date": "Date", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- Feature engineering ----------------
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_data(df, ma1=20, ma2=50):
    df = df.copy()
    df[f"MA{ma1}"] = df["Close"].rolling(ma1).mean()
    df[f"MA{ma2}"] = df["Close"].rolling(ma2).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df["Return"] = df["Close"].pct_change()
    for lag in [1,2,3,5]:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def prepare_features(df, ma1=20, ma2=50):
    features = ["Open","High","Low","Volume", f"MA{ma1}", f"MA{ma2}", "RSI", "Return", "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5"]
    X = df[features].fillna(0)
    y = df["Close"]
    return X, y

# ---------------- Model training & predict ----------------
def train_model(df, ma1=20, ma2=50):
    X, y = prepare_features(df, ma1, ma2)
    if len(X) < 20:
        raise ValueError("Not enough data for training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred))
    }
    return model, scaler, metrics

def predict_next(model, scaler, df, ma1=20, ma2=50):
    X, _ = prepare_features(df, ma1, ma2)
    last = X.iloc[-1:].values
    pred = model.predict(scaler.transform(last))
    if hasattr(pred, "__len__"):
        return float(pred[0])
    return float(pred)

# ---------------- Sidebar / Inputs ----------------
tickers, names = load_ticker_list()

st.sidebar.header("⚙️ Settings")
ticker_mode = st.sidebar.radio("Ticker input mode:", ("🔍 Search by name", "🔤 Enter custom"))
if ticker_mode == "🔍 Search by name":
    # show a searchable selectbox using display = symbol - symbol (no heavy .info calls)
    options = [t for t in tickers]
    ticker = st.sidebar.selectbox("Pick stock", options, index=options.index("AAPL") if "AAPL" in options else 0)
else:
    ticker = st.sidebar.text_input("Enter custom stock ticker (e.g., RELIANCE.NS)", "AAPL").strip().upper()

period = st.sidebar.selectbox("Period", ["1M","3M","6M","1Y","2Y","5Y"], index=3)
ma1 = st.sidebar.number_input("Short MA", value=20, min_value=5, max_value=50, step=1)
ma2 = st.sidebar.number_input("Long MA", value=50, min_value=10, max_value=200, step=5)
rsi_upper = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_lower = st.sidebar.slider("RSI Oversold", 10, 40, 30)
predict_btn = st.sidebar.button("🚀 Predict")

# ---------------- Top marquee (live prices) ----------------
marquee_tickers = ["AAPL","MSFT","TSLA","GOOGL","AMZN","RELIANCE.NS","TCS.NS"]
batch_prices = get_batch_prices(marquee_tickers)
marquee_text = "  ⚫  ".join([f"{t} (${batch_prices.get(t):.2f})" if batch_prices.get(t) is not None else f"{t} (N/A)" for t in marquee_tickers])
marquee_html = f"""
<div style="white-space: nowrap; overflow: hidden; width: 100%; background-color:#0b0b10; color:#39FF14; font-weight:bold; padding: 6px 0; border-radius:6px;">
  <div style="
    display: inline-block;
    padding-left: 100%;
    animation: marquee 28s linear infinite;
    font-family: 'Trebuchet MS', sans-serif;
    font-size: 15px;
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
st.markdown(marquee_html, unsafe_allow_html=True)

# ---------------- Main header ----------------
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0.2rem;">📈 Stock Price Predictor</h1>
    <h3 style="text-align:center; color:#FF1493; margin-top:0.1rem; font-weight:normal;">Global + Custom Stocks Analysis</h3>
    """, unsafe_allow_html=True
)

# ---------------- Predict workflow ----------------
if predict_btn:
    st.markdown(f"## 📊 Results for **{ticker}**")
    with st.spinner(f"Fetching history for {ticker}..."):
        df = fetch_stock_history(ticker, period)
    if df.empty:
        st.error(f"No historical data available for {ticker}. Try a different ticker or shorter period.")
    else:
        # process and require enough rows
        df = process_data(df, ma1, ma2)
        if df.empty or len(df) < max(ma1, ma2) + 10:
            st.error(f"Not enough data to calculate features for {ticker}. Try increasing the period or reducing MA windows.")
        else:
            # Basic fundamentals for selected ticker (on-demand, single .info call)
            fundamentals_col, chart_col = st.columns([1,3])
            with fundamentals_col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📘 Fundamentals")
                try:
                    info = yf.Ticker(ticker).info
                    market_cap = info.get("marketCap", "N/A")
                    trailing_pe = info.get("trailingPE", "N/A")
                    fifty_two_high = info.get("fiftyTwoWeekHigh", "N/A")
                    fifty_two_low = info.get("fiftyTwoWeekLow", "N/A")
                    short_name = info.get("shortName", ticker)
                    st.write(f"**{short_name}** ({ticker})")
                    st.metric("Market Cap", f"{market_cap:,}" if isinstance(market_cap, (int, np.integer)) else market_cap)
                    st.metric("PE Ratio", trailing_pe)
                    st.metric("52W High", fifty_two_high)
                    st.metric("52W Low", fifty_two_low)
                except Exception:
                    st.warning("Fundamentals not available.")
                st.markdown('</div>', unsafe_allow_html=True)

                # quick live price metric
                single_price = get_batch_prices([ticker]).get(ticker)
                if single_price is None:
                    st.metric("Live Price", "N/A")
                else:
                    st.metric("Live Price", f"${single_price:.2f}")

            # display key metrics
            with chart_col:
                col1, col2, col3 = st.columns(3)
                try:
                    last_close = float(df["Close"].iloc[-1])
                    last_volume = int(df["Volume"].iloc[-1])
                    last_rsi = float(df["RSI"].iloc[-1])
                    col1.metric("Current Price", f"${last_close:.2f}")
                    col2.metric("Volume", f"{last_volume:,}")
                    col3.metric("RSI", f"{last_rsi:.2f}")
                except Exception as e:
                    st.warning(f"⚠️ Could not display latest stats: {e}")

            # Train model
            with st.spinner("Training model..."):
                try:
                    model, scaler, metrics = train_model(df, ma1, ma2)
                except Exception as e:
                    st.error(f"Model training failed: {e}")
                    st.stop()

            st.subheader("🤖 Model Performance")
            st.write(metrics)

            # Prediction
            pred_price = predict_next(model, scaler, df, ma1, ma2)
            current_price = float(df["Close"].iloc[-1])
            change = pred_price - current_price
            pct = (change / current_price) * 100 if current_price != 0 else 0.0
            st.metric("🔮 Predicted Price", f"${pred_price:.2f}", f"{pct:.2f}%")

            # Signals: MA crossover (1 for bullish, -1 for bearish)
            df["Signal"] = np.where(df[f"MA{ma1}"] > df[f"MA{ma2}"], 1, -1)
            df["Position"] = df["Signal"].diff().fillna(0)

            # ---------------- Charts ----------------
            st.subheader("📈 Price Chart & Candlestick")
            # Matplotlib price chart (close + MAs)
            fig1, ax = plt.subplots(figsize=(10,4))
            ax.plot(df["Date"], df["Close"], label="Close Price")
            ax.plot(df["Date"], df[f"MA{ma1}"], label=f"MA{ma1}", linestyle="--")
            ax.plot(df["Date"], df[f"MA{ma2}"], label=f"MA{ma2}", linestyle=":")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            st.pyplot(fig1)

            # Plotly candlestick (interactive)
            fig2 = go.Figure(data=[go.Candlestick(x=df["Date"],
                                                  open=df["Open"],
                                                  high=df["High"],
                                                  low=df["Low"],
                                                  close=df["Close"])])
            fig2.update_layout(title=f"{ticker} Candlestick", template="plotly_dark", height=500)
            st.plotly_chart(fig2, use_container_width=True)

            # Buy / Sell signals plot
            st.subheader("📌 Buy / Sell Signals (MA Crossover)")
            fig3, ax = plt.subplots(figsize=(10,4))
            ax.plot(df["Date"], df["Close"], label="Close", linewidth=1)
            ax.plot(df["Date"], df[f"MA{ma1}"], label=f"MA{ma1}")
            ax.plot(df["Date"], df[f"MA{ma2}"], label=f"MA{ma2}")

            buys = df[df["Position"] > 0]
            sells = df[df["Position"] < 0]
            if not buys.empty:
                ax.scatter(buys["Date"], buys["Close"], marker="^", s=80, label="Buy")
            if not sells.empty:
                ax.scatter(sells["Date"], sells["Close"], marker="v", s=80, label="Sell")
            ax.legend()
            st.pyplot(fig3)

            # RSI chart
            st.subheader("📉 RSI Chart")
            fig4, ax = plt.subplots(figsize=(10,2.5))
            ax.plot(df["Date"], df["RSI"], linewidth=1)
            ax.axhline(rsi_upper, linestyle="--")
            ax.axhline(rsi_lower, linestyle="--")
            ax.set_ylim(0,100)
            st.pyplot(fig4)

            # Volume chart (bar)
            st.subheader("📊 Volume")
            fig5, ax = plt.subplots(figsize=(10,2.5))
            ax.bar(pd.to_datetime(df["Date"]), df["Volume"])
            ax.set_xlabel("Date")
            ax.set_ylabel("Volume")
            fig5.autofmt_xdate()
            st.pyplot(fig5)

            # Recent data table
            st.subheader("📋 Recent Data")
            st.dataframe(df.tail(20).reset_index(drop=True))

# ---------------- Footer ----------------
st.markdown(
    """
    <div style="text-align: center; color: pink ; padding: 10px; font-size: 14px;">
        Made by <b>Om Hela</b> (Minor in AI) IIT ROPAR • Upgraded by ChatGPT
    </div>
    """,
    unsafe_allow_html=True
)

