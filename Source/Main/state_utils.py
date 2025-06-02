import os
import json
import pickle
from datetime import datetime, timedelta
from collections import deque

STATE_FILE = "shared_state.json"
BUFFER_FILE = "price_buffer.pkl"
BUFFER_MAXLEN = 240  # 12分相当
ADX_BUFFER_FILE = "adx_buffer.pkl"

# --- 状態保存 ---
def save_state(state):
    state["last_saved"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def save_price_buffer(buffer):
    with open(BUFFER_FILE, "wb") as f:
        pickle.dump(list(buffer), f)

# --- 状態読み込み ---
def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            last_saved = datetime.fromisoformat(state.get("last_saved", "1970-01-01T00:00:00"))
            if datetime.now() - last_saved > timedelta(minutes=2):
                # 古いなら空にする
                return {"trend_init_notice": False}
            return state
    except:
        return {"trend_init_notice": False}

def load_price_buffer():
    try:
        with open(BUFFER_FILE, "rb") as f:
            buffer = pickle.load(f)
            return deque(buffer, maxlen=BUFFER_MAXLEN)
    except:
        return deque(maxlen=BUFFER_MAXLEN)

import json

def save_price_history(highs, lows, closes, filename="adx_history.json"):
    with open(filename, "w") as f:
        json.dump({
            "highs": list(highs),
            "lows": list(lows),
            "closes": list(closes),
        }, f)

from collections import deque
import os

def load_price_history(filename="adx_history.json", maxlen=240):
    highs = deque(maxlen=maxlen)
    lows = deque(maxlen=maxlen)
    closes = deque(maxlen=maxlen)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            highs.extend(data.get("highs", []))
            lows.extend(data.get("lows", []))
            closes.extend(data.get("closes", []))
    return highs, lows, closes
