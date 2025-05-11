import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def run_rule_strategies(btc_data, buy_hold,timeframe):

    # -- Technical Indicators ---
    btc_data['SMA_10'] = SMAIndicator(close=btc_data['Close'], window=10).sma_indicator()
    btc_data['SMA_50'] = SMAIndicator(close=btc_data['Close'], window=50).sma_indicator()
    btc_data['RSI'] = RSIIndicator(close=btc_data['Close'], window=14).rsi()
    bb = BollingerBands(close=btc_data['Close'], window=20)
    btc_data['BB_High'] = bb.bollinger_hband()
    btc_data['BB_Low'] = bb.bollinger_lband()

    btc_data.dropna(inplace=True)

    # -- Trading Strategy Simulations -- 
    def simulate_strategy(signals, prices, strategy_name):
        balance = 10000
        btc_held = 0
        portfolio = []

        for i in range(len(signals)):
            signal = signals[i]
            price = prices[i]
            if signal == "buy" and balance > 0:
                btc_held = balance / price
                balance = 0
            elif signal == "sell" and btc_held > 0:
                balance = btc_held * price
                btc_held = 0
            portfolio.append(balance + btc_held * price)
        
        return portfolio

    close_prices = btc_data['Close'].values
    rsi = btc_data['RSI'].values
    sma_10 = btc_data['SMA_10'].values
    sma_50 = btc_data['SMA_50'].values
    bb_high = btc_data['BB_High'].values
    bb_low = btc_data['BB_Low'].values

    # RSI Strategy
    rsi_signals = []
    if timeframe == "1w":
        rsi_buy_threshold = 45
        rsi_sell_threshold = 55
    else:
        rsi_buy_threshold = 30
        rsi_sell_threshold = 70
    
    for i in range(len(rsi)):
        if rsi[i] < rsi_buy_threshold:
            rsi_signals.append("buy")
        elif rsi[i] > rsi_sell_threshold:
            rsi_signals.append("sell")
        else:
            rsi_signals.append("hold")
    rsi_portfolio = simulate_strategy(rsi_signals, close_prices, "RSI")

    # SMA Crossover Strategy
    sma_signals = []
    for i in range(len(sma_10)):
        if sma_10[i] > sma_50[i]:
            sma_signals.append("buy")
        elif sma_10[i] < sma_50[i]:
            sma_signals.append("sell")
        else:
            sma_signals.append("hold")
    sma_portfolio = simulate_strategy(sma_signals, close_prices, "SMA")

    # Bollinger Bands Strategy
    bb_signals = []
    for i in range(len(close_prices)):
        if close_prices[i] < bb_low[i]:
            bb_signals.append("buy")
        elif close_prices[i] > bb_high[i]:
            bb_signals.append("sell")
        else:
            bb_signals.append("hold")
    bb_portfolio = simulate_strategy(bb_signals, close_prices, "BB")
    close_prices = btc_data['Close'].values


    # -- Print Final Portfolio Values -- 
    print(f"Final Portfolio Value – RSI Strategy: ${rsi_portfolio[-1]:.2f}")
    print(f"Final Portfolio Value – SMA Crossover Strategy: ${sma_portfolio[-1]:.2f}")
    print(f"Final Portfolio Value – Bollinger Bands Strategy: ${bb_portfolio[-1]:.2f}")
    print(f"Final Portfolio Value – Buy & Hold Strategy: ${buy_hold:.2f}")

    
    # -- Portfolio Visualisation -- 
    plt.figure(figsize=(14, 7))
    plt.plot(rsi_portfolio, label='RSI Agent')
    plt.plot(sma_portfolio, label='SMA Agent')
    plt.plot(bb_portfolio, label='Bollinger Bands Agent')
    plt.hlines(buy_hold, xmin=0, xmax=len(rsi_portfolio)-1, colors='black', linestyles='--', label='Buy & Hold')
    plt.title(f'Rule-Based Trading Agents vs Buy & Hold ({timeframe} timeframe)')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Return dictionary of rule-based portfolios
    return {
    "RSI": rsi_portfolio,
    "SMA Crossover": sma_portfolio,
    "Bollinger Bands": bb_portfolio
}

