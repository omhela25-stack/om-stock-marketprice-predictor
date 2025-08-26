# stockpredictor.app.py
# Rewritten full app — Plotly removed, fresh UI style, uses Alpha Vantage key provided by user.
# Data sources: Alpha Vantage (primary), yfinance (optional fallback), synthetic sample fallback.
# ML: RandomForest baseline + optional LSTM comparison if TensorFlow is available.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings("ignore")

# Optional libraries
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SK_OK = True
except Exception:
    SK_OK = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_OK = True
except Exception:
    TF_OK = False

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title="Omni Market Lab", page_icon="📊", layout="wide")

# User-supplied API key (hardcoded as requested)
ALPHA_VANTAGE_API_KEY = "LPRQX827JWWLKA4R"
AV_BASE_URL = "https://www.alphavantage.co/query"

# ------------------------- STYLES (new look) -------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(180deg, #f6fffb 0%, #f0f9ff 100%); }
        .brand {
            font-weight:800; font-size:28px;
            background: linear-gradient(90deg,#0ea5a4,#06b6d4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { color:#374151; margin-top:6px; margin-bottom:18px; }
        .side-card {
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            border: 1px solid #e6eef2; border-radius:12px; padding:14px;
        }
        .accent-btn > button {
            background: linear-gradient(90deg,#ef8d53,#ffb86b);
            color: #111827; border: none; padding: .6rem 1rem; border-radius:8px;
            font-weight:700;
        }
        .muted { color:#6b7280; font-size:14px; }
        footer, header, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div style="display:flex;align-items:baseline"><div class="brand">Omni Market Lab</div><div style="flex:1"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-source stock analysis · technicals · ML predictions</div>', unsafe_allow_html=True)

# ------------------------- UTILITIES -------------------------
def period_to_days(p):
    return {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(p,365)

@st.cache_data(ttl=300)
def fetch_alpha_vantage(ticker, outputsize="full"):
    """Fetch daily time series from Alpha Vantage (TIME_SERIES_DAILY)."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": outputsize,
        "datatype": "json"
    }
    resp = requests.get(AV_BASE_URL, params=params, timeout=30)
    data = resp.json()
    if "Error Message" in data:
        raise RuntimeError(data["Error Message"])
    # find time series key (some variants may exist)
    ts_key = next((k for k in data.keys() if "Time Series" in k), None)
    if ts_key is None:
        raise RuntimeError("Alpha Vantage returned unexpected payload.")
    df = pd.DataFrame.from_dict(data[ts_key], orient="index")
    # typical columns: '1. open','2. high','3. low','4. close','5. volume'
    df.columns = ["Open","High","Low","Close","Volume"][:len(df.columns)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().reset_index().rename(columns={"index":"Date"})
    df.attrs["source"] = "alpha_vantage"
    return df

@st.cache_data(ttl=300)
def fetch_yfinance(ticker, period="1y"):
    if not YF_OK:
        raise RuntimeError("yfinance not available")
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError("yfinance returned empty data")
    df = hist.reset_index()[["Date","Open","High","Low","Close","Volume"]]
    df.attrs["source"] = "yfinance"
    return df

def generate_sample(ticker, period):
    days = period_to_days(period)
    rng = pd.date_range(end=datetime.now(), periods=days, freq="B")
    seed = abs(hash(ticker)) % (2**32)
    rs = np.random.RandomState(seed)
    base = 120.0 if "." not in ticker else 800.0
    rets = rs.normal(0.0004, 0.02, len(rng))
    price = np.cumprod(1+rets) * base
    high = price * (1 + rs.normal(0.003, 0.008, len(price)))
    low = price * (1 - rs.normal(0.003, 0.008, len(price)))
    openp = np.r_[price[0], price[:-1] * (1 + rs.normal(0, 0.003, len(price)-1))]
    vol = (rs.lognormal(np.log(900_000), 0.9, len(price))).astype(int)
    df = pd.DataFrame({"Date":rng,"Open":openp,"High":high,"Low":low,"Close":price,"Volume":vol})
    df.attrs["source"] = "sample_data"
    return df

@st.cache_data(ttl=300)
def unified_fetch(ticker, period="1y"):
    """Try Alpha Vantage -> yfinance -> sample fallback. Trim to period."""
    # normalize some suffixes commonly used
    if ticker.endswith(".BO") or ticker.endswith(".NS"):
        ticker = ticker.split(".")[0] + ".NSE"
    # primary: AlphaVantage (gentle sleep for rate limits)
    try:
        time.sleep(1)
        df = fetch_alpha_vantage(ticker, outputsize="full")
    except Exception:
        try:
            yf_period = "5y" if period in {"2y","5y"} else "1y"
            df = fetch_yfinance(ticker, period=yf_period)
        except Exception:
            df = generate_sample(ticker, period)
    start = datetime.now() - timedelta(days=period_to_days(period))
    df = df[df["Date"] >= start].reset_index(drop=True)
    return df

def calc_rsi(close, window=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(window).mean()
    loss = (-d.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

def enrich(df):
    if df is None or df.empty: 
        return df
    z = df.copy()
    z["MA20"] = z["Close"].rolling(20).mean()
    z["MA50"] = z["Close"].rolling(50).mean()
    z["RSI"]  = calc_rsi(z["Close"])
    z["Ret"]  = z["Close"].pct_change()
    z["Vol_MA10"] = z["Volume"].rolling(10).mean()
    for k in [1,2,3,5]:
        z[f"Close_lag{k}"] = z["Close"].shift(k)
    z = z.dropna().reset_index(drop=True)
    return z

def prepare_ml(df):
    features = ["Open","High","Low","Volume","MA20","MA50","RSI","Ret","Vol_MA10"] + [f"Close_lag{k}" for k in [1,2,3,5]]
    features = [c for c in features if c in df.columns]
    X = df[features].copy()
    y = df["Close"].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cut = int(len(Xs) * 0.8)
    return Xs[:cut], Xs[cut:], y[:cut].values, y[cut:].values, scaler, features

def train_random_forest(df):
    if not SK_OK:
        return None
    Xtr, Xte, ytr, yte, scaler, feats = prepare_ml(df)
    model = RandomForestRegressor(n_estimators=140, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    y_trp = model.predict(Xtr)
    y_tep = model.predict(Xte)
    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(ytr, y_trp))),
        "test_rmse":  float(np.sqrt(mean_squared_error(yte, y_tep))),
        "train_mae":  float(np.mean(np.abs(ytr - y_trp))),
        "test_mae":   float(np.mean(np.abs(yte - y_tep))),
        "train_r2":   float(r2_score(ytr, y_trp)),
        "test_r2":    float(r2_score(yte, y_tep)),
        "train_size": int(len(Xtr)),
        "test_size":  int(len(Xte)),
    }
    fi = pd.DataFrame({"feature":feats, "importance":model.feature_importances_}).sort_values("importance", ascending=False)
    return model, scaler, metrics, fi

def predict_next(model, scaler, df):
    feats = ["Open","High","Low","Volume","MA20","MA50","RSI","Ret","Vol_MA10"] + [f"Close_lag{k}" for k in [1,2,3,5]]
    feats = [c for c in feats if c in df.columns]
    Xlast = df[feats].iloc[[-1]].values
    Xs = scaler.transform(Xlast)
    return float(model.predict(Xs)[0])

# LSTM utilities (from your original style)
def make_lstm_dataset(arr, look_back=60):
    X, y = [], []
    for i in range(len(arr)-look_back):
        X.append(arr[i:i+look_back, 0])
        y.append(arr[i+look_back, 0])
    X = np.array(X); y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ------------------------- UI / Controls -------------------------
# Left side panel with altered color scheme and layout
with st.sidebar:
    st.markdown('<div class="side-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configure Analysis")
    market_choice = st.radio("Market", ["US", "IN", "Custom"])
    if market_choice == "US":
        ticker = st.selectbox("Ticker", ["AAPL","GOOGL","MSFT","TSLA","AMZN","NVDA","META","NFLX","JPM","V"])
    elif market_choice == "IN":
        ticker = st.selectbox("Ticker", ["RELIANCE.NSE","TCS.NSE","INFY.NSE","HDFCBANK.NSE","WIPRO.NSE","ITC.NSE","SBIN.NSE"])
    else:
        ticker = st.text_input("Custom ticker", "AAPL")
    period = st.selectbox("History", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
    pred_days = st.slider("Prediction horizon (days)", 1, 14, 1)
    st.markdown("---")
    st.markdown("### 🔁 Environment")
    st.write(f"yfinance: {'✅' if YF_OK else '❌'}   scikit-learn: {'✅' if SK_OK else '❌'}   TF: {'✅' if TF_OK else '❌'}")
    st.markdown('<div class="accent-btn">', unsafe_allow_html=True)
    run = st.button("Analyze & Predict")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not run:
    st.info("Set options in the sidebar and click **Analyze & Predict**.")
    st.stop()

# ------------------------- Fetch + Enrich -------------------------
with st.spinner("Fetching data..."):
    raw = unified_fetch(ticker, period=period)
    if raw is None or raw.empty:
        st.error("No data returned. Check ticker or try another period.")
        st.stop()
    data = enrich(raw)

source = raw.attrs.get("source","unknown")
st.success(f"Loaded {len(raw)} rows for {ticker} (source: {source})")

# ------------------------- Main Tabs -------------------------
tab_overview, tab_models, tab_visuals, tab_features, tab_data = st.tabs(
    ["Overview","Models","Visuals","Feature Impact","Data"]
)

# ---------- OVERVIEW ----------
with tab_overview:
    st.subheader(f"{ticker} summary")
    last = raw["Close"].iloc[-1]
    prev = raw["Close"].iloc[-2] if len(raw)>1 else last
    delta = last - prev
    pct = delta/prev*100 if prev else 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Close", f"{last:.2f}")
    col2.metric("Change", f"{delta:.2f}", f"{pct:.2f}%")
    col3.metric("Volume", f"{int(raw['Volume'].iloc[-1]):,}")
    col4.metric("Volatility (σ)", f"{raw['Close'].pct_change().std()*100:.2f}%")
    st.markdown("**Basic Info**")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Data source: **{source}**")
        st.write(f"Rows loaded: **{len(raw):,}**")
    with c2:
        st.write(f"History window: **{period}**")
        st.write(f"Prediction horizon: **{pred_days} day(s)**")

# ---------- MODELS ----------
with tab_models:
    st.subheader("Random Forest (fast baseline)")
    if not SK_OK:
        st.warning("scikit-learn not available — cannot train Random Forest.")
    else:
        with st.spinner("Training Random Forest..."):
            rf_model, rf_scaler, rf_metrics, rf_fi = train_random_forest(data)
        if rf_model is None:
            st.error("Random Forest training failed.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Test R²", f"{rf_metrics['test_r2']:.3f}")
            m2.metric("Test RMSE", f"{rf_metrics['test_rmse']:.2f}")
            m3.metric("Test MAE", f"{rf_metrics['test_mae']:.2f}")
            # Next-day estimate
            try:
                next_est = predict_next(rf_model, rf_scaler, data)
                cur = data["Close"].iloc[-1]
                d = next_est - cur
                p = d/cur*100 if cur else 0
                st.info(f"Next est. close: **{next_est:.2f}**  (Δ {d:.2f}, {p:.2f}%)")
                if p>2: st.success("Signal: Bullish")
                elif p>0: st.info("Signal: Mild Bullish")
                elif p>-2: st.warning("Signal: Neutral")
                else: st.error("Signal: Bearish")
            except Exception as e:
                st.warning(f"Could not compute next estimate: {e}")

    st.markdown("---")
    st.subheader("Deep LSTM vs Simple Linear (optional)")
    if not TF_OK or not SK_OK:
        st.info("TensorFlow and scikit-learn are required to run deep comparisons.")
    else:
        look_back = st.slider("LSTM look-back", 30, 120, 60, 10)
        epochs = st.slider("LSTM epochs", 1, 10, 3)
        run_deep = st.checkbox("Train LSTM & compare (may take time)")
        if run_deep:
            with st.spinner("Preparing LSTM dataset..."):
                series = raw[["Close"]].astype(float)
                scaler_mm = MinMaxScaler(feature_range=(0,1))
                scaled = scaler_mm.fit_transform(series)
                X, y = make_lstm_dataset(scaled, look_back)
                cut = int(len(X)*0.8)
                Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
            with st.spinner("Training LSTM..."):
                lstm = build_lstm((Xtr.shape[1], 1))
                lstm.fit(Xtr, ytr, epochs=epochs, batch_size=32, verbose=0)
                ypred = lstm.predict(Xte, verbose=0)
                ypred_inv = scaler_mm.inverse_transform(ypred)
                yte_inv = scaler_mm.inverse_transform(yte.reshape(-1,1))
                mse_lstm = mean_squared_error(yte_inv, ypred_inv)
            # Linear baseline
            from sklearn.linear_model import LinearRegression
            Xidx = np.arange(len(scaled)).reshape(-1,1)
            Xtr_lr, Xte_lr = Xidx[:cut+look_back], Xidx[cut+look_back:]
            ytr_lr, yte_lr = scaled[:cut+look_back], scaled[cut+look_back:]
            lr = LinearRegression().fit(Xtr_lr, ytr_lr)
            ypred_lr = lr.predict(Xte_lr)
            ypred_lr_inv = scaler_mm.inverse_transform(ypred_lr)
            yte_lr_inv = scaler_mm.inverse_transform(yte_lr)
            mse_lr = mean_squared_error(yte_lr_inv, ypred_lr_inv)
            colA, colB = st.columns(2)
            with colA:
                st.metric("LSTM MSE", f"{mse_lstm:.4f}")
                fig, ax = plt.subplots()
                ax.plot(yte_inv, label="Actual", color="#0ea5a4")
                ax.plot(ypred_inv, label="LSTM", color="#ef8d53", alpha=0.9)
                ax.set_title("LSTM vs Actual")
                ax.legend()
                st.pyplot(fig)
            with colB:
                st.metric("LinearReg MSE", f"{mse_lr:.4f}")
                fig, ax = plt.subplots()
                ax.plot(yte_lr_inv, label="Actual", color="#0ea5a4")
                ax.plot(ypred_lr_inv, label="LinearReg", color="#60a5fa")
                ax.set_title("LinearReg vs Actual")
                ax.legend()
                st.pyplot(fig)

# ---------- VISUALS ----------
with tab_visuals:
    st.subheader("Price, Moving Averages & RSI")
    # Price + MAs using streamlit line chart (fast & interactive)
    plot_df = data.set_index("Date")[["Close"]].copy()
    if "MA20" in data.columns: plot_df["MA20"] = data.set_index("Date")["MA20"]
    if "MA50" in data.columns: plot_df["MA50"] = data.set_index("Date")["MA50"]
    st.line_chart(plot_df)

    st.markdown("**Volume**")
    st.bar_chart(data.set_index("Date")["Volume"])

    if "RSI" in data.columns and not data["RSI"].isna().all():
        st.markdown("**RSI**")
        fig, ax = plt.subplots(figsize=(8,2.5))
        ax.plot(data["Date"], data["RSI"], color="#06b6d4", linewidth=1.6)
        ax.axhline(70, color="#ef4444", linestyle="--", linewidth=0.9)
        ax.axhline(30, color="#10b981", linestyle="--", linewidth=0.9)
        ax.set_ylim(0,100)
        ax.set_ylabel("RSI")
        ax.set_xlabel("")
        st.pyplot(fig)

# ---------- FEATURE IMPACT ----------
with tab_features:
    st.subheader("Feature Importance (Random Forest)")
    if SK_OK:
        trained = train_random_forest(data)
        if trained is not None:
            _,_,metrics,fi = trained
            if fi is not None and not fi.empty:
                top = fi.head(12).sort_values("importance", ascending=True)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.barh(top["feature"], top["importance"], color="#4ade80")
                ax.set_xlabel("Importance")
                ax.set_title("Top features")
                st.pyplot(fig)
            col1, col2, col3 = st.columns(3)
            col1.metric("R² (test)", f"{metrics['test_r2']:.3f}")
            col2.metric("RMSE (test)", f"{metrics['test_rmse']:.2f}")
            col3.metric("MAE (test)", f"{metrics['test_mae']:.2f}")
        else:
            st.warning("Training was not successful — no feature importance to show.")
    else:
        st.info("Install scikit-learn to enable feature importance visualization.")

# ---------- DATA ----------
with tab_data:
    st.subheader("Recent Data")
    show = data.tail(80).copy()
    if "Date" in show.columns:
        show["Date"] = pd.to_datetime(show["Date"]).dt.strftime("%Y-%m-%d")
    cols = [c for c in ["Date","Open","High","Low","Close","Volume","MA20","MA50","RSI"] if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, height=420)
    csv = raw.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name=f"{ticker}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# ---------- FOOTER / DISCLAIMER ----------
st.markdown(
    """
    <div style="background:#fff8f1;border:1px solid #fde3c8;padding:12px;border-radius:10px;margin-top:12px">
    <strong>Disclaimer:</strong> This application is for educational and research purposes only. 
    Predictions are probabilistic and not financial advice. Always perform your own due diligence.
    </div>
    """,
    unsafe_allow_html=True
)
