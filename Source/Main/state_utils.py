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

def save_adx_buffers(adx):
    adx["last_saved"] = datetime.now().isoformat()
    with open(ADX_BUFFER_FILE, "w") as f:
        json.dump(adx, f)

def load_adx_buffers():
    if not os.path.exists(ADX_BUFFER_FILE):
        # print("[INFO] ADXバッファファイルが存在しません。初期化します。")
        return None

    try:
        with open(ADX_BUFFER_FILE, "r", encoding="utf-8") as f:
            adx_data = json.load(f)

        # 構造確認と初期化フォールバック
        return adx_data.get("adx", None)
            

    except Exception as e:
        #print(f"[WARN] ADXバッファ読み込みに失敗: {e}")
        return None