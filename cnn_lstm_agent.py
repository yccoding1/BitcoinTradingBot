import time
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
import joblib

def run_lstm_cnn(train_data, test_data, buy_hold, timeframe):

    if train_data.empty or test_data.empty:
        raise ValueError("Training or testing dataset is empty.")
    
    # -- Feature Engineering Function --
    def prepare_features(df):
        df = df.copy()
        df['Future Close'] = df['Close'].shift(-1)
        df['Return'] = (df['Future Close'] - df['Close']) / df['Close']
        df['SMA_10'] = df['Close'].rolling(window=10).mean().shift(1)
        df['SMA_50'] = df['Close'].rolling(window=50).mean().shift(1)
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi().shift(1)
        bb = BollingerBands(close=df['Close'], window=20)
        df['BB_High'] = bb.bollinger_hband().shift(1)
        df['BB_Low'] = bb.bollinger_lband().shift(1)
        df.dropna(inplace=True)
        return df

    train_data = prepare_features(train_data)
    test_data = prepare_features(test_data)

    features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'BB_High', 'BB_Low', 'Volume']
    target = 'Return'

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = train_data[features].values
    y_train = train_data[target].values
    X_train_scaled = scaler.fit_transform(X_train)

    X_test = test_data[features].values
    y_test = test_data[target].values
    X_test_scaled = scaler.transform(X_test)


    # -- Sequence Length -- 
    if timeframe == "1h":
        SEQ_LENGTH = 60
    elif timeframe == "4h":
        SEQ_LENGTH = 60
    elif timeframe == "1d":
        SEQ_LENGTH = 30
    elif timeframe == "1w":
        SEQ_LENGTH = 12
    else:
        SEQ_LENGTH = 110

    def create_sequences(data, targets, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(data) - seq_length):
            X_seq.append(data[i:i+seq_length])
            y_seq.append(targets[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQ_LENGTH)
    current_prices = test_data['Close'].values[-len(y_test_seq):]


    # -- Model Architecture & Training --
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQ_LENGTH, len(features))))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=64, validation_split=0.2, verbose=1)


    # -- Model Saving --
    # joblib.dump(scaler, "cnn_scaler.save")
    # model.save("cnnlstm_model.h5")


    # -- Trading Simulation --
    test_predictions = model.predict(X_test_seq).flatten()
    balance = 10000
    btc_held = 0
    STOP_LOSS = 0.95
    TAKE_PROFIT = 1.05
    portfolio_values = []

    for i in range(len(test_predictions)):
        predicted_return = test_predictions[i]
        current_price = current_prices[i]

        if predicted_return > 0 and balance > 0:
            entry_price = current_price
            btc_held = balance / entry_price
            balance = 0
        elif predicted_return < 0 and btc_held > 0:
            balance = btc_held * current_price
            btc_held = 0

        if btc_held > 0:
            if current_price < entry_price * STOP_LOSS or current_price > entry_price * TAKE_PROFIT:
                balance = btc_held * current_price
                btc_held = 0

        portfolio_values.append(balance + (btc_held * current_price))

    # -- Portfolio Visualisation --
    print(f"\nCNN-LSTM: ${portfolio_values[-1]:.2f}")
    print(f"Buy & Hold: ${buy_hold:.2f}")

    plt.figure(figsize=(12,6))
    plt.plot(portfolio_values, label='CNN-LSTM Agent')
    plt.hlines(buy_hold, 0, len(portfolio_values), colors='red', linestyles='--', label='Buy & Hold')
    plt.title(f'CNN-LSTM Agent vs Buy & Hold ({timeframe} timeframe)')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    pad_length = len(test_data) - len(portfolio_values)
    full_portfolio = [10000] * pad_length + portfolio_values


    return full_portfolio




