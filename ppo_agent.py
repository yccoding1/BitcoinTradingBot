import gym
import numpy as np
import pandas as pd
from gym import spaces
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def run_ppo_agent(train_data, trade_data, buy_hold, lookback, timeframe):


    # -- Technical Indicators for Training Data --
    train_data = train_data.copy()
    train_data['SMA_10'] = SMAIndicator(train_data['Close'], window=10).sma_indicator()
    train_data['SMA_50'] = SMAIndicator(train_data['Close'], window=50).sma_indicator()
    train_data['RSI'] = RSIIndicator(train_data['Close'], window=14).rsi()
    bb = BollingerBands(train_data['Close'], window=20)
    train_data['BB_High'] = bb.bollinger_hband()
    train_data['BB_Low'] = bb.bollinger_lband()
    train_data.dropna(inplace=True)
    train_data.reset_index(drop=True, inplace=True)


    # -- Technical Indicators for Evaluation Data --
    trade_data = trade_data.copy()
    trade_data['SMA_10'] = SMAIndicator(trade_data['Close'], window=10).sma_indicator()
    trade_data['SMA_50'] = SMAIndicator(trade_data['Close'], window=50).sma_indicator()
    trade_data['RSI'] = RSIIndicator(trade_data['Close'], window=14).rsi()
    bb_eval = BollingerBands(trade_data['Close'], window=20)
    trade_data['BB_High'] = bb_eval.bollinger_hband()
    trade_data['BB_Low'] = bb_eval.bollinger_lband()
    trade_data.dropna(inplace=True)
    trade_data.reset_index(drop=True, inplace=True)


    # -- PPO Trading Environment -- 
    class BitcoinTradingEnv(gym.Env):
        def __init__(self, df, window_size=lookback, initial_balance=10000):
            super(BitcoinTradingEnv, self).__init__()
            self.df = df
            self.window_size = window_size
            self.initial_balance = initial_balance
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(window_size, 7), dtype=np.float32
            )
            self.reset()

        def reset(self):
            self.current_step = self.window_size
            self.balance = self.initial_balance
            self.btc_held = 0
            self.total_asset = self.initial_balance
            return self._next_observation()

        def _next_observation(self):
            obs = self.df.iloc[self.current_step - self.window_size:self.current_step][
                ['Close', 'SMA_10', 'SMA_50', 'RSI', 'BB_High', 'BB_Low', 'Volume']
            ].values
            return obs.astype(np.float32)

        def step(self, action):
            current_price = self.df.iloc[self.current_step]['Close']
            prev_total_asset = self.balance + self.btc_held * current_price

            if action == 1 and self.balance > 0:
                self.btc_held = self.balance / current_price
                self.balance = 0
            elif action == 2 and self.btc_held > 0:
                self.balance = self.btc_held * current_price
                self.btc_held = 0

            self.current_step += 1
            done = self.current_step >= len(self.df) - 1
            current_asset = self.balance + self.btc_held * current_price
            reward = current_asset - prev_total_asset
            self.total_asset = current_asset
            obs = self._next_observation()
            return obs, reward, done, {}


    # -- PPO Agent Training -- 
    env = BitcoinTradingEnv(train_data, window_size=lookback)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)



    # -- Model Saving to deploy in Live-Trading -- 
    # model.save("ppo_model")
    # print("Model trained and saved.")



    # -- Evaluate PPO on Trade Data -- 
    eval_env = BitcoinTradingEnv(trade_data, window_size=lookback)
    obs = eval_env.reset()
    portfolio = []

    for _ in range(len(trade_data) - eval_env.window_size - 1):
        action, _ = model.predict(obs)
        obs, reward, done, _ = eval_env.step(action)
        portfolio.append(eval_env.total_asset)
        if done:
            break

    # Match length with trade_data by pre-filling
    pad_length = len(trade_data) - len(portfolio)
    portfolio = [10000] * pad_length + portfolio

    # -- Portfolio Visualisation --
    print(f"\nFinal PPO Portfolio Value: ${portfolio[-1]:.2f}")
    print(f"Buy & Hold Value: ${buy_hold:.2f}")

    plt.figure(figsize=(12,6))
    plt.plot(portfolio, label='PPO Agent')
    plt.hlines(buy_hold, 0, len(portfolio), colors='red', linestyles='--', label='Buy & Hold')
    plt.title(f'PPO Trading Agent vs Buy & Hold ({timeframe} timeframe)')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return portfolio
