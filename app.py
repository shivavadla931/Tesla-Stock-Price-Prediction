import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS (Light Professional Theme)
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #202121;
    color: #f0f0f0;
}
h1, h2, h3 {
    color: #f7f0f0;
}
.metric-box {
    background-color: #1130ba;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.footer {
    color: #888888;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("💹 Tesla Stock Price Prediction")
st.subheader("Deep Learning based Time-Series Forecasting")

st.markdown("""
This application demonstrates how **deep learning models** can forecast **Tesla’s future stock closing prices** using historical time-series data.
""")

# -------------------------------------------------
# TOP NAVIGATION (TABS)
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📈 Forecast Studio", "💸 Price Analytics", "🧩 Business Perspective"]
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/TESLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df[["Adj Close"]]

data = load_data()

# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

TIME_STEPS = 60

# -------------------------------------------------
# LOAD MODELS (DEPLOYMENT SAFE)
# -------------------------------------------------
rnn_model = load_model("rnn_best_model.h5", compile=False)
lstm_model = load_model("lstm_best_model.h5", compile=False)

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.title("Model Controls")

model_choice = st.sidebar.radio(
    "Select Prediction Engine",
    ["Best Model (Recommended)", "SimpleRNN", "LSTM"]
)

if model_choice == "Best Model (Recommended)":
    model = rnn_model
    selected_model_name = "SimpleRNN (Best - Lower MSE)"
elif model_choice == "SimpleRNN":
    model = rnn_model
    selected_model_name = "SimpleRNN"
else:
    model = lstm_model
    selected_model_name = "LSTM"
    
st.sidebar.markdown(
    f"""
    ### ✅ Active Model
    **{selected_model_name}**

    - Selected based on historical performance (MSE)
    """
)
st.sidebar.caption("--------")
st.sidebar.markdown("## Forecast Horizon Settings")
days_to_predict = st.sidebar.number_input(
    "Forecast Horizon (Days)",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Select how many future days to forecast"
)

# -------------------------------------------------
# PREDICTION LOGIC WITH LOADING ANIMATION
# -------------------------------------------------

last_sequence = scaled_data[-TIME_STEPS:]
current_sequence = last_sequence.reshape(1, TIME_STEPS, 1)

predictions = []

with st.spinner("Generating Forecast... ⏳"):
    for _ in range(days_to_predict):
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0, 0])

        current_sequence = np.append(
            current_sequence[:, 1:, :],
            pred.reshape(1, 1, 1),
            axis=1
        )

# Convert back to original scale
predictions = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
)

# -------------------------------------------------
# ENHANCED FORECAST METRICS
# -------------------------------------------------
last_close = data.iloc[-1, 0]
predicted_close = predictions.flatten()
predicted_open = np.insert(predicted_close[:-1], 0, last_close)

predicted_high = predicted_close * 1.02
predicted_low = predicted_close * 0.98

pct_change = ((predicted_close - predicted_open) / predicted_open) * 100

signal = []
for p in pct_change:
    if p > 1:
        signal.append("Buy")
    elif p < -1:
        signal.append("Sell")
    else:
        signal.append("Hold")
forecast_df = pd.DataFrame({
    "Predicted Open (USD)": predicted_open.round(2),
    "Predicted Close (USD)": predicted_close.round(2),
    "Expected High (USD)": predicted_high.round(2),
    "Expected Low (USD)": predicted_low.round(2),
    "Daily Change (%)": pct_change.round(2),
    "Signal": signal
})

# -------------------------------------------------
# TAB 1 — FORECAST STUDIO
# -------------------------------------------------
with tab1:
    st.markdown("## Stock Price Forecast")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"<div class='metric-box'><h3>Latest Close</h3><h2>${last_close:.2f}</h2></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='metric-box'><h3>Data Records</h3><h2>{len(data)}</h2></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            "<div class='metric-box'><h3>Prediction Engine</h3><h2>RNN / LSTM</h2></div>",
            unsafe_allow_html=True
        )

    # 🔽 EVERYTHING BELOW MUST BE INDENTED
    st.markdown("## Stock Price Forecast Summary")

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(
            f"""
            <div style="font-size:14px;color:#ccc">Next Day Close (USD)</div>
            <div style="font-size:34px;font-weight:700">${predicted_close[0]:.2f}</div>
            <div style="color:#ff4d4f;font-size:14px">↓ {pct_change[0]:.2f}%</div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div style="font-size:14px;color:#ccc">Highest Expected Price</div>
            <div style="font-size:34px;font-weight:700">${predicted_high.max():.2f}</div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        badge_color = "#fc3235" if "Sell" in signal[0] else "#18da69"
        st.markdown(
            f"""
            <div style="font-size:14px;color:#ccc">Overall Signal</div>
            <div style="font-size:30px;font-weight:700;color:{badge_color}">
                {signal[0]}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Day-wise Price Outlook")

    for i in range(len(predicted_close)):
        row = st.columns([1.3, 1.3, 1.3, 1.3, 1.3, 1.2])

        with row[0]:
            st.markdown(f"**Day {i+1} Open**<br>${predicted_open[i]:.2f}", unsafe_allow_html=True)

        with row[1]:
            st.markdown(f"**Close**<br>${predicted_close[i]:.2f}", unsafe_allow_html=True)

        with row[2]:
            st.markdown(f"**High**<br>${predicted_high[i]:.2f}", unsafe_allow_html=True)

        with row[3]:
            st.markdown(f"**Low**<br>${predicted_low[i]:.2f}", unsafe_allow_html=True)

        with row[4]:
            color = "#2ecc71" if pct_change[i] > 0 else "#ff4d4f"
            st.markdown(
                f"**Change**<br><span style='color:{color}'>{pct_change[i]:.2f}%</span>",
                unsafe_allow_html=True
            )

        with row[5]:
            btn_color = "#0d8e43" if "Buy" in signal[i] \
                        else "#fc0328" if "Sell" in signal[i] \
                        else "#f5f0f1" if "Hold" in signal[i] else "#ffffff"
            st.markdown(
                f"""
                <div style="
                    background:{btn_color};
                    color:{'#ffffff' if btn_color != '#f5f0f1' else '#000000'};
                    padding:10px;
                    border-radius:10px;
                    text-align:center;
                    font-weight:600">
                    {signal[i]}
                </div>
                """,
                unsafe_allow_html=True
            )

# -------------------------------------------------
# TAB 2 — PRICE ANALYTICS
# -------------------------------------------------
with tab2:
    st.markdown("## 📈 Price Analytics")

    st.markdown("""
    - Tesla stock exhibits **trend-driven growth**
    - Volatility makes short-term forecasting valuable
    - Deep learning captures **temporal dependencies**
    """)

    st.line_chart(data.tail(300))

    st.markdown("### Trend Projection")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(data.index[-120:], data.values[-120:], label="Actual Price", color="#1f77ff", linewidth=2.5)

    future_dates = pd.date_range(
        start=data.index[-1],
        periods=days_to_predict + 1,
        freq="B"
    )[1:]

    ax.plot(future_dates, predictions, label="Predicted Price",
            color="#ff6f61", linestyle="--", marker="o", linewidth=2.5)

    ax.axvspan(future_dates[0], future_dates[-1], color="#ffe4e1", alpha=0.6)

    ax.set_title(f"Tesla Stock Forecast using {model_choice}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (USD)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------------------------
# TAB 3 — BUSINESS PERSPECTIVE
# -------------------------------------------------
with tab3:
    st.markdown("""
    ## Business Use Case Alignment

### 📌 Trading & Investment
- Supports **buy/sell decision-making**
- Can act as an input to **algorithmic trading systems**

### 📌 Risk Management
- Helps assess **future price movements**
- Assists in portfolio rebalancing decisions

### 📌 Financial Forecasting
- Useful for **long-term investment planning**
- Enables data-driven financial insights

### 📌 Research & Innovation
- Demonstrates comparison between **SimpleRNN and LSTM**
- Can be extended with **news sentiment, macroeconomic data, or advanced models**

### 📌 Interpretation Notes
- **Predicted Close** is generated by the deep learning model (SimpleRNN / LSTM).
- **Predicted Open, High, and Low** are estimated using historical price behavior.
- **Signal** is a rule-based indicator derived from predicted price movement.
""")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(" © 2026 • Built with Streamlit & Deep Learning Models ")

st.caption(" Developed by Vadla Shiva Kumar ")
