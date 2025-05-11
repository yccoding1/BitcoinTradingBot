# BitcoinTradingBot

# Multi-Agent Bitcoin Trading System

An AI-powered trading bot that compares five different strategies for Bitcoin (BTC/USDT) trading:  
- CNN-LSTM (deep learning),  
- PPO (reinforcement learning),  
- RSI (momentum-based rule),  
- SMA Crossover (trend-based rule),  
- Bollinger Bands (volatility-based rule).

## Features
- Backtesting with historical Binance OHLCV data
- Live trading simulation with real-time price feed
- Modular architecture for easily plugging in new strategies
- Logs and visualisations for detailed performance comparison

## Prerequisites

- **Python 3.9** ([Download](https://www.python.org/downloads/))
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Disk Space**: ~1GB free space
- **Internet**: Required for fetching live Binance data

**For GPU Acceleration**:
- **NVIDIA GPU** (Windows): CUDA-enabled GPU with [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive) and [cuDNN 8.1](https://developer.nvidia.com/cudnn)
- **Apple Silicon** (macOS): M1/M2/M3 chip recommended

## Installation

### **Windows Setup**

**Create virtual environment**
```bash
python -m venv trading_env
trading_env\Scripts\activate
```

**Install dependencies**
```bash
pip install tensorflow==2.10.0
pip install pandas numpy matplotlib scikit-learn ta ccxt stable-baselines3 gym joblib
```

**Verify installation**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```


### **MacOS Setup**
**Install Miniforge (Apple Silicon)**
```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o Miniforge3.sh
bash Miniforge3.sh
```

**Create environment**
```bash
conda create -n trading_env python=3.9
conda activate trading_env
```

**Install dependencies**
```bash
conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.10.0 tensorflow-metal==0.6.0
pip install pandas numpy matplotlib scikit-learn ta ccxt stable-baselines3 gym joblib
```

**Verify GPU acceleration**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Usage

# Windows
**Running Backtest:**
```bash
trading_env\Scripts\activate
python backtest_agents.py 
```

**Running Live Trading:**
```bash
trading_env\Scripts\activate
python live_trading.py 
```

# macOS
**Running Backtest:**
```bash
conda activate trading_env
python backtest_agents.py
```

**Running Live Trading:**
```bash
conda activate trading_env #(if not yet in trading_env)
python live_trading.py
```

## Notes
* CNN-LSTM model is retrained during each backtest run if joblib.dump(scaler, "cnn_scaler.save") and model.save("cnnlstm_model.h5") is not commented out
* PPO is trained using a custom OpenAI Gym environment.
* Technical indicators include RSI, SMA, and Bollinger Bands from ta library.
* Buy & Hold baseline is included in every comparison in historical backtesting. 
* To simulate realistic volatility responses, Bollinger Bands and RSI window sizes are adjusted dynamically.

## Disclaimer
This project is for educational purposes only. It does not constitute financial advice, and live trading involves risk. Backtest performance does not guarantee future results.







