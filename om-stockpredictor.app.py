# stockpredictor.app.py
# Merged & refactored: keeps your LSTM/LR + adds full analysis stack, multi-source data, indicators,
# model comparison, feature importance, richer charts, downloads, API status, and a fresh UI layout.
# Sources merged: your original stockpredictor.app.py and the provided app.py (features & flows).
# stockpredictor.app.py (Plotly-free version)
# stockpredictor.app.py
# Plotly-free merged app

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# --- Optional deps ---
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

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="Quantum Stocks Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- API setup ----
AV_API_KEY = "LPRQX827JWWLKA4R"
AV_BASE_URL = "https://www.alphavantage.co/query"

# ---- Style ----
st.markdown("""
<style>
:root { --pri:#6e56cf; --pri2:#22c55e; --bg:#0b0d12; --panel:#121521; --muted:#9aa0b4; --line:#23263a; }
.stApp { background: radial-gradient(1200px 500px at 10% -10%, rgba(110,86,207,.15), transparent), var(--bg); }
h1.title {
  font-size: 2.2rem; font-weight: 800;
  background: linear-gradient(90deg, var(--pri), var(--pri2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Quantum Stocks Lab</h1>', unsafe_allow_html=True)

# ------------------- Helper Data -------------------
def _period_to_days(p):
    return {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(p,365)

@st.cache_data(ttl=300)
def fetch_alpha_vantage(ticker, outputsize="full"):
    params = {"function":"TIME_SERIES_DAILY","symbol":ticker,"apikey":AV_API_KEY,
              "outputsize":outputsize,"datatype":"json"}
    r = requests.get(AV_BASE_URL, params=params, timeout=30)
    data = r.json()
    key = next((k for k in data.keys() if "Time Series" in k), None)
    if not key: raise RuntimeError("No data")
    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.columns = ["Open","High","Low","Close","Volume"][: len(df.columns)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().reset_index().rename(columns={"index":"Date"})
    df.attrs["source"] = "alpha_vantage"
    return df

@st.cache_data(ttl=300)
def fetch_yfinance(ticker, period="1y"):
    if not YF_OK: raise RuntimeError("yfinance not available")
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty: raise RuntimeError("empty yf data")
    df = hist.reset_index()[["Date","Open","High","Low","Close","Volume"]]
    df.attrs["source"] = "yfinance"
    return df

def gen_sample_data(ticker, period):
    days = _period_to_days(period)
    rng = pd.date_range(end=datetime.now(), periods=days, freq="B")
    rs = np.random.RandomState(42)
    price = np.cumprod(1+rs.normal(0.0005,0.02,days))*100
    high = price*(1+rs.normal(0.005,0.01,days))
    low  = price*(1-rs.normal(0.005,0.01,days))
    openp= np.r_[price[0], price[:-1]*(1+rs.normal(0,0.003,days-1))]
    vol  = rs.lognormal(12,0.5,days).astype(int)
    df = pd.DataFrame({"Date":rng,"Open":openp,"High":high,"Low":low,"Close":price,"Volume":vol})
    df.attrs["source"] = "sample_data"
    return df

@st.cache_data(ttl=300)
def fetch_unified(ticker, period="1y"):
    try:
        df = fetch_alpha_vantage(ticker)
    except:
        try:
            df = fetch_yfinance(ticker, period)
        except:
            df = gen_sample_data(ticker, period)
    start = datetime.now() - timedelta(days=_period_to_days(period))
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

# ------------------- ML -------------------
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
    fi = pd.Series(model.feature_importances_, index=feats)
    return model,rmse,yte,ypred,fi

# ------------------- LSTM helper -------------------
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
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ------------------- Sidebar -------------------
ticker=st.sidebar.text_input("Ticker","AAPL")
period=st.sidebar.selectbox("Period",["1mo","3mo","6mo","1y","2y","5y"],3)
run_btn=st.sidebar.button("🚀 Run Analysis")

if not run_btn:
    st.info("Choose ticker and click Run.")
    st.stop()

# ------------------- Main Action -------------------
raw=fetch_unified(ticker,period)
data=enrich(raw)

st.success(f"Loaded {len(raw)} rows via {raw.attrs.get('source','unknown')}")

tab_overview, tab_models, tab_charts, tab_table = st.tabs(
    ["Overview 📋","Models 🤖","Charts 📈","Data 🔎"]
)

# ---- OVERVIEW ----
with tab_overview:
    st.metric("Last Close", f"{raw['Close'].iloc[-1]:.2f}")
    st.metric("Volume", f"{int(raw['Volume'].iloc[-1]):,}")

# ---- MODELS ----
with tab_models:
    if SK_OK:
        model,rmse,ytrue,ypred,fi = train_random_forest(data)
        st.write(f"Random Forest Test RMSE: {rmse:.2f}")
        fig,ax=plt.subplots()
        fi.sort_values().plot.barh(ax=ax)
        st.pyplot(fig)
    if TF_OK:
        look_back=60
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        series=raw[["Close"]].values.astype(float)
        scaled=scaler.fit_transform(series)
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
with tab_table:
    st.dataframe(data.tail(60))
    st.download_button("Download CSV", raw.to_csv(index=False), file_name=f"{ticker}.csv")
