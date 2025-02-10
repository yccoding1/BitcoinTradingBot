# BitcoinTradingBot

# Bitcoin Trading Bot with LSTM Prediction

An AI-powered trading bot that predicts Bitcoin price movements using an LSTM model and simulates trading strategies.

## Prerequisites

- **Python 3.9** ([Download](https://www.python.org/downloads/))
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Disk Space**: 1GB free space

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
pip install pandas numpy matplotlib scikit-learn yfinance ta
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
pip install pandas numpy matplotlib scikit-learn yfinance ta
```

**Verify GPU acceleration**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Usage
**Clone the repository:**
```bash
git clone https://github.com/yourusername/bitcoin-trading-bot.git
cd bitcoin-trading-bot
```

**Run the bot:**
# Windows
```bash
trading_env\Scripts\activate
python BTC_TradingBot.py
```

# macOS
```bash
conda activate trading_env
python3 BTC_TradingBot.py
```




