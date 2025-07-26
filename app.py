# 📦 Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# 🌐 Streamlit page configuration
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

# 🌗 Theme-aware color configuration
def get_theme_colors():
    theme = st.get_option("theme.base")
    if theme == "dark":
        return {
            "bg_color": "#0E1117",
            "text_color": "#FAFAFA",
            "line_color": "#FF9800",
            "grid_color": "#444"
        }
    else:
        return {
            "bg_color": "#FFFFFF",
            "text_color": "#000000",
            "line_color": "#FF5733",
            "grid_color": "#DDD"
        }

colors = get_theme_colors()

# 🎯 App Title (theme-compatible)
title_color = "#FAFAFA" if st.get_option("theme.base") == "dark" else "#2C3E50"
st.markdown(f"<h1 style='text-align: center; color: {title_color}; font-size: 42px;'>💹 Stock Price Predictor App</h1>", unsafe_allow_html=True)
st.markdown("---")

# 📌 User Instruction Note
st.markdown(f"""
<div style='background-color:#FFF3CD;padding:15px;border-left:6px solid #FFC107;border-radius:5px;margin-bottom:15px; color:#000000;'>
    <strong>📌 Note:</strong> Please enter the <strong>stock ticker symbol</strong> (e.g., <code>GOOG</code> for Google, <code>AAPL</code> for Apple, <code>TSLA</code> for Tesla).<br>
    Do <strong>not</strong> enter full company names.<br><br>
    📊 You can find ticker symbols at <a href='https://finance.yahoo.com/' target='_blank'><strong>Yahoo Finance</strong></a> – the data in this app is fetched from there.
</div>
""", unsafe_allow_html=True)

# 🔤 Stock ticker input
stock = st.text_input("🔎 Enter Stock Ticker Symbol", "GOOG")

# 📅 Fetch stock data from Yahoo Finance
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start=start, end=end)

# 🧠 Load trained model
model = load_model("SPP.keras")

# 📃 Show data sample
st.markdown(f"### 🧾 Historical Stock Data: `{stock}`")
st.dataframe(google_data.tail(), use_container_width=True)

# ➕ Calculate moving averages
google_data['MA_for_250_days'] = google_data['Close'].rolling(window=250).mean()
google_data['MA_for_200_days'] = google_data['Close'].rolling(window=200).mean()
google_data['MA_for_100_days'] = google_data['Close'].rolling(window=100).mean()

# 📈 Universal theme-adaptive plotting function
def plot_graph(figsize, values, full_data, overlay=False, overlay_data=None, title=""):
    fig, ax = plt.subplots(figsize=figsize, facecolor=colors["bg_color"])
    ax.plot(values, label='Target', color=colors["line_color"], linewidth=2)
    ax.plot(full_data['Close'], label='Close Price', color='#3498DB', alpha=0.3)
    if overlay and overlay_data is not None:
        ax.plot(overlay_data, label='Overlay', color='limegreen', linestyle='--')

    ax.set_xlabel("Date", fontsize=12, color=colors["text_color"])
    ax.set_ylabel("Price (USD)", fontsize=12, color=colors["text_color"])
    ax.set_title(title, fontsize=16, fontweight='bold', color=colors["text_color"])
    ax.tick_params(axis='x', colors=colors["text_color"])
    ax.tick_params(axis='y', colors=colors["text_color"])
    ax.grid(True, alpha=0.3, color=colors["grid_color"])
    ax.legend()
    return fig

# 📊 Moving Averages & Trends
st.markdown("### 📊 Moving Averages & Trend Visualization")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_graph((20, 6), google_data['MA_for_250_days'], google_data, title='250-Day Moving Average'))

with col2:
    st.pyplot(plot_graph((20, 6), google_data['MA_for_200_days'], google_data, title='200-Day Moving Average'))

st.pyplot(plot_graph((20, 6), google_data['MA_for_100_days'], google_data, title='100-Day Moving Average'))
st.pyplot(plot_graph((20, 6), google_data['MA_for_100_days'], google_data, overlay=True, overlay_data=google_data['MA_for_250_days'], title='Overlay: 100-Day vs 250-Day MA'))

# 📦 Prepare test data
splitting_len = int(len(google_data) * 0.7)
x_test_df = google_data[['Close']].iloc[splitting_len:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test_df)

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
x_data, y_data = np.array(x_data), np.array(y_data)

# 🔮 Predict and inverse transform
predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# 📊 Create prediction DataFrame
ploting_data = pd.DataFrame({
    'Actual Price': inv_y_test.reshape(-1),
    'Predicted Price': inv_predictions.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

# 📄 Show predictions table
st.markdown("### 📋 Prediction Comparison Table")
st.dataframe(ploting_data.tail(), use_container_width=True)

# 📈 Final Actual vs Predicted plot
st.markdown("### 📈 Full Timeline: Actual vs Predicted Close Price")

fig, ax = plt.subplots(figsize=(22, 7), facecolor=colors["bg_color"])
ax.plot(google_data['Close'][:splitting_len + 100], label="Training Data", color='gray')
ax.plot(ploting_data['Actual Price'], label="Actual Test Data", color='#2980B9')
ax.plot(ploting_data['Predicted Price'], label="Predicted Test Data", color='#E67E22')
ax.set_title("📈 Complete Stock Price Timeline with Predictions", fontsize=18, fontweight='bold', color=colors["text_color"])
ax.set_xlabel("Date", fontsize=12, color=colors["text_color"])
ax.set_ylabel("Price (USD)", fontsize=12, color=colors["text_color"])
ax.tick_params(axis='x', colors=colors["text_color"])
ax.tick_params(axis='y', colors=colors["text_color"])
ax.grid(True, alpha=0.3, color=colors["grid_color"])
ax.legend()
st.pyplot(fig)

# 📏 RMSE display
rmse = np.sqrt(np.mean((inv_predictions - inv_y_test) ** 2))
st.markdown(f"<h4 style='text-align:center; color:#28A745;'>✅ Model RMSE: <code>{rmse:.2f} USD</code></h4>", unsafe_allow_html=True)