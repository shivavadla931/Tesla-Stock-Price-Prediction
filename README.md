# Tesla Stock Price Prediction — Deep Learning Forecasting Web App
A fully interactive Streamlit-based web application that predicts Tesla (TSLA) stock prices using advanced Deep Learning models (SimpleRNN & LSTM).
This project demonstrates end-to-end time-series forecasting using real historical data, feature scaling, model training, and ultra-smooth UI visualizations optimized for both web & mobile.
## **Live Demo**

🔗 https://teslastockprediction.streamlit.app/

## **Project Overview**

This application forecasts Tesla’s stock price for the next 1–10 days using two neural network architectures:

- SimpleRNN – lightweight, fast, good for short-term trends
- LSTM – deeper memory capability, better for long-range patterns

A "Best Model" option intelligently selects the most accurate model based on historical MSE performance.
The app also generates:
- Predicted Open, Close, High, Low
- % change for each day
- Automatic Buy / Sell / Hold signal
- Trend visualization charts
- Business interpretation insights

## **Key Features**

✔ **Real-time Price Forecasting**

Predicts future Tesla stock prices using trained deep learning models.

✔ **Multiple Model Options**

Users can switch between:
- SimpleRNN
- LSTM
- Best Model (Recommended)
  
✔ **Loading Animation**

Displays a smooth progress spinner during prediction generation.

✔ **Clean, Interactive UI**

- Tab-based layout
- Card-style metric visualization
- Mobile-responsive design

✔ **Day-wise Results**

Beautiful card-based display showing:
- Opening price
- Closing price
- High/Low
- Percentage change
- Market signal indicator

✔ **Business Insights**

Built-in explanation of:
- Trading use cases
- Risk management
- Portfolio decisions

## **Tech Stack**

| Component        | Technology           |
|------------------|----------------------|
| Frontend         | Streamlit            |
| Backend          | Python               |
| ML/DL Framework  | TensorFlow / Keras   |
| Data Processing  | NumPy, Pandas        |
| Visualization    | Matplotlib, Streamlit Charts |
| Deployment       | Streamlit Cloud      |

## **Project Structure**
```
tesla-stock-price-prediction
│
├── app.py
├── requirements.txt
├── data/
│   └── TESLA.csv
├── rnn_best_model.h5
│── lstm_best_model.h5
```

## **Deployment (Streamlit Cloud)**

1. Push all files to GitHub
2. Go to https://share.streamlit.io
3. Click New App
4. Connect GitHub repo
5. Set:
- Main file: app.py
- Python version: 3.10
6. Deploy!

## **Model Training Summary**

Both models were trained using:

- 60-step sliding window
- MinMax scaling
- RNN/LSTM layers
- Dense output layer
- EarlyStopping callback
- ModelCheckpoint to save the best model

## **Disclaimer**
This project is meant for educational and research purposes only.
It should not be used as financial or trading advice.
