# stock_predictor_app.py
# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Predictor")
st.write("Enter a ticker and select a period in the sidebar, then click 🚀 Predict to fetch data and train the model.")

# Sidebar
ticker = st.sidebar.text_input("Enter stock ticker", "AAPL")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
predict_btn = st.sidebar.button("🚀 Predict")

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
            # --- your existing metrics, model, charts ---
            # (no change here, just leave as you had it)
# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Predictor")
st.write("Enter a ticker and select a period in the sidebar, then click 🚀 Predict to fetch data and train the model.")

# Sidebar
ticker = st.sidebar.text_input("Enter stock ticker", "AAPL")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
predict_btn = st.sidebar.button("🚀 Predict")

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
            # --- your existing metrics, model, charts ---
            # (no change here, just leave as you had it)















