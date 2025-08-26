# om-stockpredictor.app.py
# Clean version without Plotly, with your Alpha Vantage API key

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings("ignore")

# --- Optional dependencies ---
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

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Omni Market Lab", page_icon="📊", layout="wide")

# Hardcoded Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "LPRQX827JWWLKA4R"
AV_BASE_URL = "https://www.alphavantage.co/query"

# ---------------- STYLE ----------------
st.markdown(
    """
    <style>
        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(180deg, #f6fffb 0%, #f0f9ff 100%); }
        .brand {
            font-weight:800; font-size:28px;
            background: linear-gradient(90deg,#0ea5a4,#06b6d4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { color:#374151; margin-top:6px; margin-bottom:18px; }
        footer, header, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="brand">Omni Market Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Stock analysis · technicals · ML predictions</div>', unsafe_allow_html=True)

# ---------------- DATA HELPERS ----------------
def period_to_days(p):
    return {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(p,365)

@st.cache_data(ttl=300)
def fetch_alpha_vantage(ticker, outputsize="full"):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": outputsize,
        "datatype": "json"
    }
    r = requests.get(AV_BASE_URL, params=params, timeout=30)
    data = r.json()
    ts_key = next((k for k in data.keys() if "Time Series" in k), None)
    if ts_key is None:
        raise RuntimeError("No time series data")
    df = pd.DataFrame.from_dict(data[ts_key], orient="index")
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
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise RuntimeError("yfinance returned no data")
    df = hist.reset_index()[["Date","Open","High","Low","Close","Volume"]]
    df.attrs["source"] = "yfinance"
    return df

def generate_sample(ticker, period):
    days = period_to_days(period)
    rng = pd.date_range(end=datetime.now(), periods=days, freq="B")
    rs = np.random.RandomState(abs(hash(ticker)) % (2**32))
    price = np.cumprod(1+rs.normal(0.0004,0.02,days))*100
    high = price*(1+rs.normal(0.005,0.01,days))
    low  = price*(1-rs.normal(0.005,0.01,days))
    openp= np.r_[price[0], price[:-1]*(1+rs.normal(0,0.003,days-1))]
    vol  = rs.lognormal(12,0.5,days).astype(int)
    df = pd.DataFrame({"Date":rng,"Open":openp,"High":high,"Low":low,"Close":price,"Volume":vol})
    df.attrs["source"] = "sample_data"
    return df

@st.cache_data(ttl=300)
def unified_fetch(ticker, period="1y"):
    try:
        time.sleep(1)
        df = fetch_alpha_vantage(ticker)
    except Exception:
        try:
            df = fetch_yfinance(ticker, period)
        except Exception:
            df = generate_sample(ticker, period)
    start = datetime.now() - timedelta(days=period_to_days(period))
    return df[df["Date"]>=start].reset_index(drop=True)

def calc_rsi(close, window=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(window).mean()
    loss = (-d.clip(upper=0)).rolling(window).mean()
    rs = gain/loss
    return 100 - (100/(1+rs))

def enrich(df):
    df["MA20"]=df["Close"].rolling(20).mean()
    df["MA50"]=df["Close"].rolling(50).mean()
    df["RSI"]=calc_rsi(df["Close"])
    return df

# ---------------- ML HELPERS ----------------
def train_random_forest(df):
    if not SK_OK: return None
    feats=["Open","High","Low","Volume","MA20","MA50","RSI"]
    df=df.dropna()
    X,y=df[feats],df["Close"]
    n=int(len(X)*0.8)
    Xtr,Xte,ytr,yte=X.iloc[:n],X.iloc[n:],y.iloc[:n],y.iloc[n:]
    scaler=StandardScaler().fit(Xtr)
    Xtr,Xte=scaler.transform(Xtr),scaler.transform(Xte)
    model=RandomForestRegressor(n_estimators=100).fit(Xtr,ytr)
    ypred=model.predict(Xte)
    rmse=np.sqrt(mean_squared_error(yte,ypred))
    fi=pd.Series(model.feature_importances_,index=feats)
    return model,rmse,yte,ypred,fi

# LSTM utilities
def make_lstm_dataset(arr, look_back=60):
    X,y=[],[]
    for i in range(len(arr)-look_back):
        X.append(arr[i:i+look_back,0])
        y.append(arr[i+look_back,0])
    X=np.array(X); y=np.array(y)
    X=X.reshape((X.shape[0],X.shape[1],1))
    return X,y

def build_lstm(input_shape):
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam",loss="mean_squared_error")
    return model

# ---------------- SIDEBAR ----------------
ticker=st.sidebar.text_input("Ticker","AAPL")
period=st.sidebar.selectbox("Period",["1mo","3mo","6mo","1y","2y","5y"],3)
run=st.sidebar.button("🚀 Run Analysis")

if not run:
    st.info("Enter ticker and press Run.")
    st.stop()

# ---------------- MAIN ----------------
raw=unified_fetch(ticker,period)
data=enrich(raw)

st.success(f"Loaded {len(raw)} rows (source: {raw.attrs.get('source')} )")

tab_overview, tab_models, tab_charts, tab_data = st.tabs(
    ["Overview","Models","Charts","Data"]
)

# ---- OVERVIEW ----
with tab_overview:
    st.metric("Last Close", f"{raw['Close'].iloc[-1]:.2f}")
    st.metric("Volume", f"{int(raw['Volume'].iloc[-1]):,}")

# ---- MODELS ----
with tab_models:
    if SK_OK:
        model,rmse,ytrue,ypred,fi=train_random_forest(data)
        st.write(f"Random Forest RMSE: {rmse:.2f}")
        fig,ax=plt.subplots()
        fi.sort_values().plot.barh(ax=ax)
        st.pyplot(fig)
    if TF_OK:
        from sklearn.preprocessing import MinMaxScaler
        series=raw[["Close"]].values.astype(float)
        scaler=MinMaxScaler()
        scaled=scaler.fit_transform(series)
        look_back=60
        X,y=make_lstm_dataset(scaled,look_back)
        n=int(len(X)*0.8)
        Xtr,Xte,ytr,yte=X[:n],X[n:],y[:n],y[n:]
        lstm=build_lstm((look_back,1))
        lstm.fit(Xtr,ytr,epochs=1,batch_size=32,verbose=0)
        ypred=lstm.predict(Xte,verbose=0)
        ypred=scaler.inverse_transform(ypred)
        yte=scaler.inverse_transform(yte.reshape(-1,1))
        fig,ax=plt.subplots()
        ax.plot(yte,label="True")
        ax.plot(ypred,label="LSTM")
        ax.legend()
        st.pyplot(fig)

# ---- CHARTS ----
with tab_charts:
    st.line_chart(data.set_index("Date")[["Close","MA20","MA50"]])
    st.bar_chart(data.set_index("Date")["Volume"])
    if "RSI" in data:
        fig,ax=plt.subplots()
        ax.plot(data["Date"],data["RSI"],label="RSI",color="green")
        ax.axhline(70,color="red",linestyle="--")
        ax.axhline(30,color="blue",linestyle="--")
        ax.legend()
        st.pyplot(fig)

# ---- DATA ----
with tab_data:
    st.dataframe(data.tail(50))
    st.download_button("Download CSV", raw.to_csv(index=False), file_name=f"{ticker}.csv")
