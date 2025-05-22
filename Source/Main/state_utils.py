import os
import json
import pickle
from datetime import datetime, timedelta
from collections import deque

STATE_FILE = "shared_state.json"
BUFFER_FILE = "price_buffer.pkl"
BUFFER_MAXLEN = 240  # 12分相当

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
