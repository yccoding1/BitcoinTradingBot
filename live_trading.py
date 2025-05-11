import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from stable_baselines3 import PPO
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import atexit
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib


# -- Logging Setup --
log_filename = f"live_trading_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_filename)
file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging.info("=== Multi-Agent Live Trading Started ===")

# -- User Prompt --
timeframe = "5m"
allocations = {}
for agent in ["PPO", "SMA", "RSI", "Bollinger", "CNNLSTM"]:
    while True:
        try:
            alloc = float(input(f"USD to allocate to {agent} agent: "))
            allocations[agent] = alloc
            break
        except:
            print("Invalid input. Enter a number.")

REAL_TRADING = False
symbol = "BTC/USDT"
window_size = 60
model_path = "ppo_model.zip"

# -- Model Loading --
# PPO Model
model = PPO.load(model_path)

# Load CNN-LSTM model
cnn_lstm_model = load_model("cnnlstm_model.h5")

# Define scaler and feature list (must match training)
scaler = MinMaxScaler(feature_range=(0, 1))
cnn_features = ['Close', 'SMA_3', 'SMA_9', 'RSI', 'BB_High', 'BB_Low', 'Volume']
SEQ_LENGTH = 60  # or 30 if you're on 1d, adjust as needed

# -- Exchange Setup --
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
    **({'apiKey': 'XXX', 'secret': 'YYY'} if REAL_TRADING else {})
})

# -- Balances and Stats --
agents = {
    "PPO": {"usd": allocations["PPO"], "btc": 0, "buy": 0, "sell": 0},
    "SMA": {"usd": allocations["SMA"], "btc": 0, "buy": 0, "sell": 0},
    "RSI": {"usd": allocations["RSI"], "btc": 0, "buy": 0, "sell": 0},
    "Bollinger": {"usd": allocations["Bollinger"], "btc": 0, "buy": 0, "sell": 0},
    "CNNLSTM": {"usd": allocations.get("CNNLSTM", 0), "btc": 0, "buy": 0, "sell": 0},
}
last_price = 0

# -- Summary report on exit --
def generate_summary():
    logging.info("======== TRADING SUMMARY ========")
    for name, data in agents.items():
        total_value = data['usd'] + data['btc'] * last_price
        logging.info(f"{name} Agent | USD: ${data['usd']:.2f}, BTC: {data['btc']:.6f}, "
                     f"Buys: {data['buy']}, Sells: {data['sell']}, Total Value: ${total_value:.2f}")
    logging.info(f"Last BTC Price: ${last_price:.2f}")
    logging.info("=================================")

atexit.register(generate_summary)

# -- Strategy Logic --
def get_rule_signal(agent_name, row):
    '''
    if agent_name == "SMA":
        return "buy" if row['SMA_10'] > row['SMA_50'] else "sell" if row['SMA_10'] < row['SMA_50'] else "hold"
    '''
    if agent_name == "SMA":
        return "buy" if row['SMA_3'] > row['SMA_9'] else "sell" if row['SMA_3'] < row['SMA_9'] else "hold"
    if agent_name == "RSI":
        return "buy" if row['RSI'] < 30 else "sell" if row['RSI'] > 70 else "hold"
    if agent_name == "Bollinger":
        tolerance = 0.001  # 0.1% buffer
        if row['Close'] < row['BB_Low'] * (1 + tolerance):
            return "buy"
        elif row['Close'] > row['BB_High'] * (1 - tolerance):
            return "sell"
        else:
            return "hold"

def execute_trade(agent_name, agent, signal, price):
    if signal == "buy" and agent["usd"] >= 10:
        amount = agent["usd"] / price
        agent["btc"] += amount
        agent["usd"] = 0
        agent["buy"] += 1
        logging.info(f"{agent_name} Agent | BOUGHT at ${price:.2f} | New Balance – USD: ${agent['usd']:.2f}, BTC: {agent['btc']:.6f}")

    elif signal == "sell" and agent["btc"] > 0:
        agent["usd"] += agent["btc"] * price
        agent["btc"] = 0
        agent["sell"] += 1
        logging.info(f"{agent_name} Agent | SOLD at ${price:.2f} | New Balance – USD: ${agent['usd']:.2f}, BTC: {agent['btc']:.6f}")

# -- Live Loop --
while True:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=window_size + 50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # -- Optimized for 5m timeframe --
        df['SMA_3'] = SMAIndicator(df['Close'], window=5).sma_indicator()
        df['SMA_9'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['RSI'] = RSIIndicator(df['Close'], window=7).rsi()  # Faster RSI
        bb = BollingerBands(df['Close'], window=5)  # Tighter bands for quick volatility
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()

        df.dropna(inplace=True)
        logging.info(f"Fetched {len(df)} rows | Last: {df['timestamp'].iloc[-1]}")

        if len(df) >= window_size:
            obs_df = df.iloc[-window_size:][['Close', 'SMA_3', 'SMA_9', 'RSI', 'BB_High', 'BB_Low', 'Volume']]
            obs = obs_df.values.astype(np.float32).reshape(1, window_size, 7)
            current_price = df['Close'].iloc[-1]
            last_price = current_price

            # PPO Agent
            ppo_action, _ = model.predict(obs)
            ppo_signal = ["hold", "buy", "sell"][ppo_action[0]]
            execute_trade("PPO", agents["PPO"], ppo_signal, current_price)
            logging.info(f"PPO Agent | Signal: {ppo_signal.upper()}")

            # Rule-Based Agents
            for name in ["SMA", "RSI", "Bollinger"]:
                signal = get_rule_signal(name, df.iloc[-1])
                execute_trade(name, agents[name], signal, current_price)
                logging.info(f"{name} Agent | Signal: {signal.upper()}")

            # CNN-LSTM Agent

            # Prepare CNN-LSTM input
            cnn_df = df[-(SEQ_LENGTH + 50):].copy()


            # Feature engineering (same as during training)
            cnn_df['SMA_3'] = SMAIndicator(cnn_df['Close'], window=3).sma_indicator()
            cnn_df['SMA_9'] = SMAIndicator(cnn_df['Close'], window=9).sma_indicator()
            cnn_df['RSI'] = RSIIndicator(cnn_df['Close'], window=14).rsi()
            bb = BollingerBands(close=cnn_df['Close'], window=20)
            cnn_df['BB_High'] = bb.bollinger_hband()
            cnn_df['BB_Low'] = bb.bollinger_lband()
            cnn_df.dropna(inplace=True)


            if len(cnn_df) >= SEQ_LENGTH:
                try:
                    X_live = cnn_df[cnn_features].values
                    X_live_scaled = scaler.fit_transform(X_live)  
                    seq_input = X_live_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, len(cnn_features))

                    predicted_return = cnn_lstm_model.predict(seq_input, verbose=0)[0][0]
                    current_price = cnn_df['Close'].iloc[-1]
                    last_price = current_price

                    # Generate signal
                    if predicted_return > 0 and agents["CNNLSTM"]["usd"] >= 10:
                        cnn_signal = "buy"
                    elif predicted_return < 0 and agents["CNNLSTM"]["btc"] > 0:
                        cnn_signal = "sell"
                    else:
                        cnn_signal = "hold"

                    execute_trade("CNNLSTM", agents["CNNLSTM"], cnn_signal, current_price)
                    logging.info(f"CNNLSTM Agent | Signal: {cnn_signal.upper()}")

                except Exception as cnn_error:
                    logging.error(f"CNNLSTM Agent | Error during prediction: {cnn_error}")
            else:
                logging.info("CNNLSTM Agent | Skipping prediction: not enough data after dropna")

        else:
            logging.info("Waiting for more candles...")

    except Exception as e:
        logging.error(f"Exception: {e}")

    time.sleep(60)



