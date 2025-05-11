from datetime import date
import time
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from cnn_lstm_agent import run_lstm_cnn
from ppo_agent import run_ppo_agent
from rule_based_agents import run_rule_strategies
from datetime import date, datetime


def fetch_binance_data(symbol="BTC/USDT", timeframe="4h", limit=2000, max_retries=5):
    exchange = ccxt.binance()
    retries = 0

    while retries < max_retries:
        try:
            print(f"Fetching data (attempt {retries+1}/{max_retries})..")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=2000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            if not df.empty:
                print("Data successfully fetched.")
                return df
        except Exception as e:
            print(f"Error fetching data: {e}")

        retries += 1
        time.sleep(2 ** retries)

    raise ValueError("Failed to fetch data after multiple retries.")


def calculate_buy_and_hold(close_prices):
    return (10000 / close_prices[0]) * close_prices[-1]


def calculate_metrics(portfolio):
    portfolio = np.array(portfolio)
    portfolio = portfolio[~np.isnan(portfolio)]  # Remove NaNs

    if len(portfolio) < 2:
        return 0, 0, 0, 0

    returns = np.diff(portfolio) / portfolio[:-1]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

    max_drawdown = 0
    peak = portfolio[0]
    for value in portfolio:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    trade_count = 0
    win_count = 0
    for i in range(1, len(portfolio)):
        if portfolio[i] != portfolio[i-1]:
            trade_count += 1
            if portfolio[i] > portfolio[i-1]:
                win_count += 1

    win_rate = (win_count / trade_count) if trade_count > 0 else 0
    return sharpe, max_drawdown, win_rate, trade_count



# -- Run Each Agent and Collect Portfolios -- 
if __name__ == "__main__":
    # Prompt user for timeframe
    valid_timeframes = ["15m","1h", "4h", "1d", "1w"]
    while True:
        user_tf = input(f"Select timeframe ({', '.join(valid_timeframes)}): ").strip()
        if user_tf in valid_timeframes:
            timeframe = user_tf
            break
        else:
            print("Invalid choice. Please try again.")

    print(f"\nFetching Binance data for {timeframe} timeframe...")

    btc_data_all = fetch_binance_data(timeframe=timeframe)
    print(f"Fetched {len(btc_data_all)} candles for timeframe {timeframe}")

    btc_data_all = btc_data_all.sort_index()

 
    # -- Data Preparation -- 
    print(f"Data fetched successfully from {btc_data_all.index.min()} to {btc_data_all.index.max()}.\n")

    train_ratio = 0.7  # 70% for training
    split_idx = int(len(btc_data_all) * train_ratio)

    if split_idx <= 0 or split_idx >= len(btc_data_all):
        raise ValueError(f"Invalid split index {split_idx} for data length {len(btc_data_all)}")

    btc_train_data = btc_data_all.iloc[:split_idx].copy()
    btc_trade_data = btc_data_all.iloc[split_idx:].copy()

    print(f"Training data: {btc_train_data.index.min()} to {btc_train_data.index.max()}")
    print(f"Trading data: {btc_trade_data.index.min()} to {btc_trade_data.index.max()}")
    print(f"Training samples: {len(btc_train_data)}, Trading samples: {len(btc_trade_data)}")

    if len(btc_train_data) < 100 or len(btc_trade_data) < 50:
        raise ValueError(f"Insufficient data: train={len(btc_train_data)}, trade={len(btc_trade_data)}")


    # Determine minimum lookback based on timeframe
    lookback_map = {"1h": 24, "4h": 14, "1d": 30, "1w": 12}
    lookback = lookback_map.get(timeframe, 60)

    btc_data = btc_trade_data.iloc[lookback:].copy()

    # Calculate aligned Buy & Hold from lookback onwards
    close_prices = btc_data['Close'].values
    buy_hold_value = (10000 / close_prices[0]) * close_prices[-1]


    # Run agents with shared data
    print("Running CNN-LSTM Agent...")
    lstm_portfolio = run_lstm_cnn(btc_train_data, btc_trade_data, buy_hold_value, timeframe)

    print("\nRunning PPO Agent...")
    ppo_portfolio = run_ppo_agent(btc_train_data, btc_trade_data, buy_hold_value, lookback, timeframe)

    print("\nRunning Rule-Based Strategies...")
    rules_portfolios = run_rule_strategies(btc_trade_data, buy_hold_value, timeframe)


# -- Calculate Metrics --
results = []

def log_and_store(name, portfolio, bh_value, timeframe):
    sharpe, max_dd, win_rate, trades = calculate_metrics(np.array(portfolio))
    final_val = portfolio[-1]
    print(f"\n{name} ({timeframe}):")
    print(f"  Final Portfolio Value: ${final_val:.2f}")
    print(f"  Buy & Hold Value     : ${bh_value:.2f}")
    print(f"  Sharpe Ratio         : {sharpe:.2f}")
    print(f"  Max Drawdown         : {max_dd:.2%}")
    print(f"  Win Rate             : {win_rate:.2%}")
    print(f"  Trade Count          : {trades}")
    print(f"Buy & Hold starts at {btc_data.index[0]} with price ${close_prices[0]:.2f}")
    results.append({
        "Date": str(date.today()),
        "Agent": name,
        "Timeframe": timeframe,
        "Final Portfolio Value": final_val,
        "Buy & Hold": bh_value,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Trade Count": trades
    })


log_and_store("CNN-LSTM Agent", lstm_portfolio, buy_hold_value, timeframe)
log_and_store("PPO Agent", ppo_portfolio, buy_hold_value, timeframe)
for name, portfolio in rules_portfolios.items():
    log_and_store(f"{name} Strategy", portfolio, buy_hold_value, timeframe)

print("--- End ---\n")


# -- Save to CSV -- 
csv_path = "strategy_metrics_summary.csv"

# If the CSV exists, append new results; otherwise, create it
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    combined_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
else:
    combined_df = pd.DataFrame(results)

combined_df.to_csv(csv_path, index=False, float_format="%.4f")
print("\nMetrics added to 'strategy_metrics_summary.csv'.")



# -- Line Chart Visualising Portfolio Change -- 
plt.figure(figsize=(14, 8))

# CNN-LSTM
plt.plot(lstm_portfolio, label='CNN-LSTM Agent', linewidth=2)

# PPO
plt.plot(ppo_portfolio, label='PPO Agent', linewidth=2)

# Rules
for name, portfolio in rules_portfolios.items():
    plt.plot(portfolio, label=f'{name} Strategy Agent', linestyle='--', linewidth=2)

# Buy & Hold as horizontal lines for reference
plt.hlines(buy_hold_value, 0, len(ppo_portfolio), colors='brown', linestyles=':', label='Buy & Hold')

plt.title(f'Trading Strategies Comparison ({timeframe} timeframe)')
plt.xlabel('Time Steps') 
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





