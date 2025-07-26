
# 📈 Stock Price Predictor App

![Repo size](https://img.shields.io/github/repo-size/sg2499/Stock-Price-Predictor)
![Stars](https://img.shields.io/github/stars/sg2499/Stock-Price-Predictor?style=social)
![Last Commit](https://img.shields.io/github/last-commit/sg2499/Stock-Price-Predictor)
![Built with Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)

This repository contains a **Stock Price Prediction Web App** built using **LSTM**, **Streamlit**, and **Yahoo Finance API**.  
It allows users to visualize a company's stock data, view moving averages, and predict future prices using a trained deep learning model.

---

## 📸 UI Screenshot

<img src="SPP App.png" width="100%" alt="Campus Placement Prediction Form UI"/>

---

## 📁 Project Folder Structure

```
📦 Stock-Price-Predictor/
├── app.py                        # Streamlit app with enhanced UI
├── SPP.keras                     # Trained LSTM model for stock price prediction
├── requirements.txt             # Required Python libraries
├── Stock Price Prediction New.ipynb  # Full notebook for EDA, preprocessing & model training
├── assets/
│   └── stock_price_app_screenshot.png  # App screenshot image
├── README.md                    # Project documentation
```

---

## 🎯 Key Features

- 🧠 **LSTM-based Price Prediction**  
  Predicts stock closing prices using a 2-layer LSTM model trained on adjusted close prices.

- 📊 **Interactive Visualization**  
  View historical price trends, moving averages (100, 200, 250 days), and percentage change graphs.

- 💼 **Clean & Professional UI**  
  Developed with Streamlit, styled with custom headers, banners, and responsive design.

- 🔍 **Ticker-Based Search**  
  Enter any valid stock ticker (e.g., `GOOG`, `AAPL`, `MSFT`) and visualize + predict its prices.

- 📈 **Side-by-Side Comparison**  
  Compare actual vs predicted stock prices interactively.

---

## 🧪 Data Source

- ✅ Yahoo Finance via [`yfinance`](https://pypi.org/project/yfinance/) Python library.
- The app fetches **20+ years** of historical stock data (Open, High, Low, Close, Volume).

---

## ⚙️ Model Architecture

- Built using **TensorFlow Keras**.
- Model Layers:
  - `LSTM(128, return_sequences=True)`
  - `LSTM(64)`
  - `Dense(25)`
  - `Dense(1)`
- Trained on a **70:30 train-test split** using `Adj Close` values.
- Scaled with `MinMaxScaler` for better convergence.
- Final RMSE: **~2.57**

---

## 💻 Setup Instructions

### 🔧 Clone the Repository

```bash
git clone https://github.com/sg2499/Stock-Price-Predictor.git
cd Stock-Price-Predictor
```

### 🐍 Create a Virtual Environment

```bash
conda create -n stock_app python=3.11
conda activate stock_app
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 Run the App

```bash
streamlit run app.py
```

---

## 📝 Requirements

Key libraries used:

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `yfinance`
- `tensorflow`

Full list available in [`requirements.txt`](./requirements.txt)

---

## 🧠 How It Works

1. User enters a valid stock **ticker symbol**.
2. App fetches 20 years of stock data using `yfinance`.
3. Displays:
   - Full price chart
   - Moving Averages (100/200/250 days)
   - % change graphs
4. LSTM model makes predictions on the recent trend.
5. Final graph shows predicted vs original values.

---

## 📝 Note for Users

📌 **Enter only the stock ticker symbol** (e.g., `GOOG`, `AAPL`, `TSLA`).  
Do not enter full company names.  
Find ticker symbols here: [Yahoo Finance](https://finance.yahoo.com/lookup)

---

## 📬 Author

Created with 💙 by **Shailesh Gupta**  
🔗 GitHub: [sg2499](https://github.com/sg2499)  
📩 Email: shaileshgupta841@gmail.com  

> Powered by LSTM · Delivered with Streamlit · Inspired by Markets 📊
