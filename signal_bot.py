import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from tensorflow.keras.models import load_model
import requests
import time
import os
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

SIGNAL_PATH = "last_signal.txt"
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
           "AVAXUSDT", "TONUSDT", "LINKUSDT", "BCHUSDT", "APTUSDT"]
session = HTTP()
model = load_model("model.h5")

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Ошибка Telegram:", e)

def analyze_symbol(symbol):
    try:
        candles = session.get_kline(category="linear", symbol=symbol, interval="15", limit=100)['result']['list']
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df[::-1]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["rsi"] = RSIIndicator(df["close"]).rsi()
        df["ema"] = EMAIndicator(df["close"], window=100).ema_indicator()
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df.dropna(inplace=True)
        last = df.iloc[-20:]
        features = last[["open", "high", "low", "close", "volume", "rsi", "macd", "ema"]].values
        features = (features - features.min(0)) / (features.max(0) - features.min(0))
        X = np.expand_dims(features, axis=0)
        pred = model.predict(X)
        label = np.argmax(pred)
        return label, df["close"].iloc[-1]
    except Exception as e:
        print(f"Ошибка по {symbol}:", e)
        return 0, None

# Telegram test
send_telegram("🤖 Bot Started")

if os.path.exists(SIGNAL_PATH):
    with open(SIGNAL_PATH, "r") as f:
        last_signals = dict(line.strip().split(":") for line in f if ":" in line)
else:
    last_signals = {}

new_signals = {}
for sym in symbols:
    signal, price = analyze_symbol(sym)
    last_signal = int(last_signals.get(sym, -1))
    if signal != 0 and signal != last_signal:
        direction = "🟢 LONG" if signal == 1 else "🔴 SHORT"
        msg = f"{direction}\nSymbol: {sym}\nPrice: {price:.2f}\nTimeframe: 15m"
        send_telegram(msg)
        new_signals[sym] = str(signal)
        print("Отправлен сигнал:", msg)
    else:
        new_signals[sym] = str(last_signal)

with open(SIGNAL_PATH, "w") as f:
    for sym, sig in new_signals.items():
        f.write(f"{sym}:{sig}\n")
