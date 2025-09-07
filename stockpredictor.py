import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# Your ML functions (already implemented in your codebase)
# from your_model_file import train_model, predict_next

# ---------------------------------
# 📊 Sidebar Controls
# ---------------------------------
st.sidebar.header("⚙️ Settings")

# Stock selection
stock = st.sidebar.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "TSLA"])

# Timeframe selection (mapped to Alpha Vantage 'outputsize')
timeframe = st.sidebar.radio("Timeframe", ["1M", "3M", "6M", "1Y"])

# Moving averages
ma1 = st.sidebar.number_input("Short MA", value=20, min_value=5, max_value=50, step=1)
ma2 = st.sidebar.number_input("Long MA", value=50, min_value=10, max_value=200, step=5)

# RSI thresholds
rsi_upper = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_lower = st.sidebar.slider("RSI Oversold", 10, 40, 30)

# ---------------------------------
# 📥 Fetch Stock Data (Alpha Vantage)
# ---------------------------------
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # <-- replace with your key
ts = TimeSeries(key=API_KEY, output_format="pandas")

# Fetch daily data
data, meta = ts.get_daily(symbol=stock, outputsize="full")
df = data.rename(columns={
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume"
}).reset_index().rename(columns={"date": "Date"})

# Filter timeframe
if timeframe == "1M":
    df = df.tail(22)
elif timeframe == "3M":
    df = df.tail(66)
elif timeframe == "6M":
    df = df.tail(132)
elif timeframe == "1Y":
    df = df.tail(252)

df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------
# 📈 Indicators
# ---------------------------------
df[f"MA{ma1}"] = df["Close"].rolling(ma1).mean()
df[f"MA{ma2}"] = df["Close"].rolling(ma2).mean()

# RSI calculation
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# ---------------------------------
# 🏷️ Dashboard Title
# ---------------------------------
st.title(f"📊 Stock Prediction Dashboard - {stock}")
st.caption(f"Timeframe: {timeframe}")

# ---------------------------------
# 📌 Metrics
# ---------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
col3.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

# ---------------------------------
# 🤖 Model Training & Prediction
# ---------------------------------
with st.spinner("Training model..."):
    model, scaler, metrics = train_model(df)

st.subheader("🤖 Model Performance")
st.write(metrics)

pred_price = predict_next(model, scaler, df)
current_price = df["Close"].iloc[-1]
change = pred_price - current_price
pct = change / current_price * 100

st.subheader("🔮 Next Day Prediction")
st.metric("Predicted Price", f"${pred_price:.2f}", f"{pct:.2f}%")

# ---------------------------------
# 📊 Charts
# ---------------------------------
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
ax.axhline(rsi_upper, linestyle="--", color="orange", label="Overbought")
ax.axhline(rsi_lower, linestyle="--", color="green", label="Oversold")
ax.set_ylim(0, 100)
ax.legend()
st.pyplot(fig)

# ---------------------------------
# 📋 Data Table
# ---------------------------------
st.subheader("📋 Recent Data")
st.dataframe(df.tail(20))
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# Your ML functions (already implemented in your codebase)
# from your_model_file import train_model, predict_next

# ---------------------------------
# 📊 Sidebar Controls
# ---------------------------------
st.sidebar.header("⚙️ Settings")

# Stock selection
stock = st.sidebar.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "TSLA"])

# Timeframe selection (mapped to Alpha Vantage 'outputsize')
timeframe = st.sidebar.radio("Timeframe", ["1M", "3M", "6M", "1Y"])

# Moving averages
ma1 = st.sidebar.number_input("Short MA", value=20, min_value=5, max_value=50, step=1)
ma2 = st.sidebar.number_input("Long MA", value=50, min_value=10, max_value=200, step=5)

# RSI thresholds
rsi_upper = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_lower = st.sidebar.slider("RSI Oversold", 10, 40, 30)

# ---------------------------------
# 📥 Fetch Stock Data (Alpha Vantage)
# ---------------------------------
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # <-- replace with your key
ts = TimeSeries(key=API_KEY, output_format="pandas")

# Fetch daily data
data, meta = ts.get_daily(symbol=stock, outputsize="full")
df = data.rename(columns={
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume"
}).reset_index().rename(columns={"date": "Date"})

# Filter timeframe
if timeframe == "1M":
    df = df.tail(22)
elif timeframe == "3M":
    df = df.tail(66)
elif timeframe == "6M":
    df = df.tail(132)
elif timeframe == "1Y":
    df = df.tail(252)

df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------
# 📈 Indicators
# ---------------------------------
df[f"MA{ma1}"] = df["Close"].rolling(ma1).mean()
df[f"MA{ma2}"] = df["Close"].rolling(ma2).mean()

# RSI calculation
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# ---------------------------------
# 🏷️ Dashboard Title
# ---------------------------------
st.title(f"📊 Stock Prediction Dashboard - {stock}")
st.caption(f"Timeframe: {timeframe}")

# ---------------------------------
# 📌 Metrics
# ---------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
col3.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

# ---------------------------------
# 🤖 Model Training & Prediction
# ---------------------------------
with st.spinner("Training model..."):
    model, scaler, metrics = train_model(df)

st.subheader("🤖 Model Performance")
st.write(metrics)

pred_price = predict_next(model, scaler, df)
current_price = df["Close"].iloc[-1]
change = pred_price - current_price
pct = change / current_price * 100

st.subheader("🔮 Next Day Prediction")
st.metric("Predicted Price", f"${pred_price:.2f}", f"{pct:.2f}%")

# ---------------------------------
# 📊 Charts
# ---------------------------------
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
ax.axhline(rsi_upper, linestyle="--", color="orange", label="Overbought")
ax.axhline(rsi_lower, linestyle="--", color="green", label="Oversold")
ax.set_ylim(0, 100)
ax.legend()
st.pyplot(fig)

# ---------------------------------
# 📋 Data Table
# ---------------------------------
st.subheader("📋 Recent Data")
st.dataframe(df.tail(20))
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# Your ML functions (already implemented in your codebase)
# from your_model_file import train_model, predict_next

# ---------------------------------
# 📊 Sidebar Controls
# ---------------------------------
st.sidebar.header("⚙️ Settings")

# Stock selection
stock = st.sidebar.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "TSLA"])

# Timeframe selection (mapped to Alpha Vantage 'outputsize')
timeframe = st.sidebar.radio("Timeframe", ["1M", "3M", "6M", "1Y"])

# Moving averages
ma1 = st.sidebar.number_input("Short MA", value=20, min_value=5, max_value=50, step=1)
ma2 = st.sidebar.number_input("Long MA", value=50, min_value=10, max_value=200, step=5)

# RSI thresholds
rsi_upper = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_lower = st.sidebar.slider("RSI Oversold", 10, 40, 30)

# ---------------------------------
# 📥 Fetch Stock Data (Alpha Vantage)
# ---------------------------------
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # <-- replace with your key
ts = TimeSeries(key=API_KEY, output_format="pandas")

# Fetch daily data
data, meta = ts.get_daily(symbol=stock, outputsize="full")
df = data.rename(columns={
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume"
}).reset_index().rename(columns={"date": "Date"})

# Filter timeframe
if timeframe == "1M":
    df = df.tail(22)
elif timeframe == "3M":
    df = df.tail(66)
elif timeframe == "6M":
    df = df.tail(132)
elif timeframe == "1Y":
    df = df.tail(252)

df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------
# 📈 Indicators
# ---------------------------------
df[f"MA{ma1}"] = df["Close"].rolling(ma1).mean()
df[f"MA{ma2}"] = df["Close"].rolling(ma2).mean()

# RSI calculation
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# ---------------------------------
# 🏷️ Dashboard Title
# ---------------------------------
st.title(f"📊 Stock Prediction Dashboard - {stock}")
st.caption(f"Timeframe: {timeframe}")

# ---------------------------------
# 📌 Metrics
# ---------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
col3.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

# ---------------------------------
# 🤖 Model Training & Prediction
# ---------------------------------
with st.spinner("Training model..."):
    model, scaler, metrics = train_model(df)

st.subheader("🤖 Model Performance")
st.write(metrics)

pred_price = predict_next(model, scaler, df)
current_price = df["Close"].iloc[-1]
change = pred_price - current_price
pct = change / current_price * 100

st.subheader("🔮 Next Day Prediction")
st.metric("Predicted Price", f"${pred_price:.2f}", f"{pct:.2f}%")

# ---------------------------------
# 📊 Charts
# ---------------------------------
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
ax.axhline(rsi_upper, linestyle="--", color="orange", label="Overbought")
ax.axhline(rsi_lower, linestyle="--", color="green", label="Oversold")
ax.set_ylim(0, 100)
ax.legend()
st.pyplot(fig)

# ---------------------------------
# 📋 Data Table
# ---------------------------------
st.subheader("📋 Recent Data")
st.dataframe(df.tail(20))
















