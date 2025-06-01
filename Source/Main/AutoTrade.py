# Copyright (c) 2025 Innovation Craft Inc. All Rights Reserved.
# 本ソフトウェアは Innovation Craft Inc. のプロプライエタリライセンスに基づいて提供されています。
# 本ソフトウェアの使用、複製、改変、再配布には Innovation Craft Inc. の事前の書面による許可が必要です。

import os
import hmac
import hashlib
import json
import requests
import time
import csv
import logging
from datetime import datetime
from dotenv import load_dotenv
from slack_notify import notify_slack
import sys
import asyncio
import statistics
import pandas as pd
import statistics
import signal
from collections import deque
import mysql.connector
from conf_load import load_settings_from_db
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from state_utils import (
    save_state,
    load_state,
    save_price_buffer,
    load_price_buffer,
)
from Price import extract_price_from_response
from logs import write_log
shared_state = {
    "trend": None,
    "last_trend": None,
    "trend_init_notice": False,
    "last_margin_ratio": None,
    "last_margin_notify": None,
    "margin_alert_sent": False,
    "last_short_ma": None,  # ← これを追加
    "last_long_ma": None ,
    "last_skip_notice": None,
    "last_spread":None,  # ← これも追加
    "rsi_adx_none_notice":False,
    "RSI":None,
    "entry_time":None,
    "loss_streak":None,
    "cooldown_until":None,
    "vstop_active":False
}

args=sys.argv
file_path = sys.argv[0]  # スクリプトファイルのパス
folder_path = os.path.dirname(os.path.abspath(file_path))
os.chdir(folder_path)

TEST = False # デバッグ用フラグ
spread_history = deque(maxlen=5)


def calc_macd(close_prices, short_period=12, long_period=26, signal_period=9):
    #MACDとシグナルラインを返す
    close_series = pd.Series(close_prices)
    ema_short = close_series.ewm(span=short_period).mean()
    ema_long = close_series.ewm(span=long_period).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_period).mean()
    return macd.tolist(), signal.tolist()

# ===ログ設定 ===
LOG_FILE1 = "fx_debug_log.txt"
_log_last_reset = datetime.now()

def setup_logging():
    """初期ログ設定（起動時）"""
    handler = TimedRotatingFileHandler(
        LOG_FILE1,
        when='midnight',       # 毎日深夜にローテート
        interval=1,            # 1日ごとにローテート
        backupCount=7,         # 最大7個のバックアップファイルを保持
        encoding='utf-8',      # エンコーディング指定
        utc=False              # 日本時間でのローテーション
    )

    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )


# 最大240本まで保持（例：1分足で4時間分）
price_history = deque(maxlen=240)
def validate_candle_shape_from_prices(price_history, trend):
    """
    終値の履歴からローソク足を自動算出し、ノイズ足（ヒゲが長く実体が短い）を判定する。
    
    Parameters:
    - price_history: list or deque of float
        終値の履歴（少なくとも2つ以上必要）
    - trend: str ("BUY" or "SELL")

    Returns:
    - trend: str または None（ノイズ足であればNone）
    """

    if len(price_history) < 2:
        return trend  # データ不足なら判定しない

    # 直近1本のローソク足を生成（過去4データから）
    # 終値: 現在
    current_close = price_history[-1]
    # 始値: 1本前の足の終値（または指定の範囲平均でもOK）
    current_open = price_history[-2]
    # 高値・安値: 過去数本で計算（ここでは直近4本）
    recent_range = price_history[-4:] if len(price_history) >= 4 else price_history
    current_high = max(recent_range)
    current_low = min(recent_range)

    # 実体・ヒゲ長を計算
    real_body = abs(current_close - current_open)
    upper_wick = current_high - max(current_close, current_open)
    lower_wick = min(current_close, current_open) - current_low

    # ノイズ判定（実体が短く、ヒゲが相対的に長い）
    if trend == "BUY":
        if real_body < 0.03 and lower_wick > real_body * 2:
            notify_slack(f"ノイズ判定: BUYエントリー見送り（実体={real_body:.4f}, 下ヒゲ={lower_wick:.4f}）")
            return None

    elif trend == "SELL":
        if real_body < 0.03 and upper_wick > real_body * 2:
            notify_slack(f"ノイズ判定: SELLエントリー見送り（実体={real_body:.4f}, 上ヒゲ={upper_wick:.4f}）")
            return None

    return trend  # 問題なければそのまま返す

try:
    setup_logging()
except Exception as e:
    print(f"ログ初期化時にエラー: {e}")
notify_slack("自動売買システム起動")

# == 記録済みデータ読み込み ===
shared_state = load_state()
price_buffer = load_price_buffer()

LOG_FILE = "fx_trade_log.csv"
LOSS_STREAK_THRESHOLD = 3
COOLDOWN_DURATION_SEC = 180  # 3分間

DEFAULT_CONFIG = {
    "LOT_SIZE": 1000,
    "MAX_SPREAD": 0.03,
    "MAX_LOSS": 20,
    "MIN_PROFIT": 40,
    "CHECK_INTERVAL": 3,
    "MAINTENANCE_MARGIN_RATIO": 0.5,
    "VOL_THRESHOLD": 0.03
}

macd_valid = False
macd_reason = ""

def record_result(profit, shared_state):
    if profit < 0:
        shared_state["loss_streak"] = shared_state.get("loss_streak", 0) + 1
        if shared_state["loss_streak"] >= LOSS_STREAK_THRESHOLD:
            shared_state["cooldown_until"] = time.time() + COOLDOWN_DURATION_SEC
            notify_slack(f"[連敗クールダウン] {LOSS_STREAK_THRESHOLD}連敗のため{COOLDOWN_DURATION_SEC//60}分間停止")
    else:
        shared_state["loss_streak"] = 0  # 勝てばリセット

def is_in_cooldown(shared_state):
    cooldown_until = shared_state.get("cooldown_until", 0)
    return time.time() < cooldown_until, max(0, int(cooldown_until - time.time()))

async def monitor_hold_status(shared_state, stop_event, interval_sec=1):
    last_notified = {}  # 建玉ごとの通知済みprofit記録

    while not stop_event.is_set():
        positions = get_positions()
        prices = get_price()
        if prices is None:
            await asyncio.sleep(interval_sec)
            continue

        ask = prices["ask"]
        bid = prices["bid"]

        for pos in positions:
            pid = pos["positionId"]
            entry = float(pos["price"])
            size = int(pos["size"])
            side = pos.get("side", "BUY").upper()

            profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

            # 通知条件：利益または損失が±10円以上、かつ通知内容が前回と違うとき
            if abs(profit) > 10:
                prev = last_notified.get(pid)
                if prev is None or abs(prev - profit) >= 5:  # 5円以上変化時のみ再通知
                    notify_slack(f"[保有] 建玉{pid} 継続中: {profit}円")
                    last_notified[pid] = profit
        await asyncio.sleep(interval_sec)

def load_config_from_mysql():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor()
        cursor.execute("SELECT `key`, `value` FROM bot_config")
        rows = cursor.fetchall()
        config = DEFAULT_CONFIG.copy()
        for key, value in rows:
            if key in config:
                # 自動型変換
                try:
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                except:
                    pass
        cursor.close()
        conn.close()
        return config
    except Exception as e:
        print(f"⚠️ 設定読み込み失敗（MySQL）：{e}")
        return DEFAULT_CONFIG


# == 損益即時監視用タスク ==
async def monitor_positions_fast(shared_state, stop_event, interval_sec=1):
    SLIPPAGE_BUFFER = 5  # 許容スリッページ（円）
    
    while not stop_event.is_set():
        positions = get_positions()
        prices = get_price()
        if prices is None:
            await asyncio.sleep(interval_sec)
            continue

        ask = prices["ask"]
        bid = prices["bid"]

        for pos in positions:
            entry = float(pos["price"])
            pid = pos["positionId"]
            size_str = int(pos["size"])
            side = pos.get("side", "BUY").upper()
            close_side = "SELL" if side == "BUY" else "BUY"

            profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

            # スリッページバッファ込みで早めに判断
            if profit <= (-MAX_LOSS + SLIPPAGE_BUFFER):
                notify_slack(f"[即時損切] 損失が {profit} 円（許容: -{MAX_LOSS}円 ±{SLIPPAGE_BUFFER}）→ 強制決済実行")
                
                start = time.time()
                close_order(pid, size_str, close_side)
                end = time.time()
                
                record_result(profit, shared_state)
                write_log("LOSS_CUT_FAST", bid)
                
                # 遅延ログも記録
                elapsed = end - start
                if elapsed > 0.5:
                    logging.warning(f"[遅延警告] 決済リクエストに {elapsed:.2f} 秒かかりました")

                shared_state["trend"] = None
                shared_state["last_trend"] = None
                shared_state["entry_time"] = time.time()

        await asyncio.sleep(interval_sec)

# === 設定読み込み ===
config = load_config_from_mysql()
SYMBOL="USD_JPY"
LOT_SIZE = config["LOT_SIZE"]
MAX_SPREAD = config["MAX_SPREAD"]
MAX_LOSS = config["MAX_LOSS"]
MIN_PROFIT = config["MIN_PROFIT"]
CHECK_INTERVAL = config["CHECK_INTERVAL"]
MAINTENANCE_MARGIN_RATIO = config["MAINTENANCE_MARGIN_RATIO"]
VOL_THRESHOLD = config["VOL_THRESHOLD"]

def is_high_volatility(prices, threshold=VOL_THRESHOLD):
    if len(prices) < 5:
        return False
    return statistics.stdev(prices[-5:]) > threshold

def handle_exit(signum, frame):
    print("SIGTERM 受信 → 状態保存")
    save_state(shared_state)
    save_price_buffer(price_buffer)
    sys.exit(0)

# === 環境変数の読み込み ===
conf=load_settings_from_db()
API_KEY = conf["API_KEY"]
API_SECRET = conf["API_SECRET"]
BASE_URL_FX = "https://forex-api.coin.z.com/private"
FOREX_PUBLIC_API = "https://forex-api.coin.z.com/public"

# === トレンド判定関数 ===
signal.signal(signal.SIGTERM, handle_exit)

# === 現在価格取得 ===
def get_price():
    try:
        res = requests.get(f"{FOREX_PUBLIC_API}/v1/ticker")
        res.raise_for_status()
        data = res.json().get("data", [])
        for item in data:
            if item.get("symbol") == SYMBOL:
                return {"ask": float(item["ask"]), "bid": float(item["bid"])}
        logging.error(f"[価格] 指定シンボル {SYMBOL} が見つかりません")
        return None
    except Exception as e:
        logging.error(f"[価格] 取得失敗: {e}")
        return None

# === RSIを計算 ===
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    prices_series = pd.Series(prices)
    delta = prices_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# === ADXを計算 ===
def calculate_adx(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return None

    highs = pd.Series(highs)
    lows = pd.Series(lows)
    closes = pd.Series(closes)

    plus_dm = highs.diff()
    minus_dm = lows.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = highs - lows
    tr2 = (highs - closes.shift()).abs()
    tr3 = (lows - closes.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # ✅ 分母がゼロのとき小さな値に置き換える
    denominator = plus_di + minus_di
    denominator = denominator.replace(0, 1e-10)

    dx = (abs(plus_di - minus_di) / denominator) * 100
    adx = dx.rolling(window=period).mean()
    
    result = adx.iloc[-1]
    return None if pd.isna(result) else result

macd_valid = False

# === トレンド判定を拡張（RSI+ADX込み） ===
async def monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=None):
    import statistics
    from collections import deque
    from datetime import datetime
    import time

    price_buffer = deque(maxlen=240)
    high_prices = deque(maxlen=240)
    low_prices = deque(maxlen=240)
    close_prices = deque(maxlen=240)

    last_rsi_state = None
    last_adx_state = None
    
    sstop = 0
    while not stop_event.is_set():
        if is_market_open() != "OPEN":
                if sstop==0:
                    notify_slack(f"[市場] 市場がCLOSEかメンテナンス中")
                    sstop = 1    
                continue
        else:
            sstop = 0
        in_cd, remaining = is_in_cooldown(shared_state)
        if in_cd:
            notify_slack(f"[クールダウン中] あと{remaining}秒 → エントリー判断を停止中")
            await asyncio.sleep(interval_sec)
            continue

        prices = get_price()
        now = datetime.now()
        if now.hour >= 22:
            if not shared_state.get("vstop_active", False):
                notify_slack(f"[クールダウン] 22時以降のため自動売買スキップ")
                shared_state["vstop_active"] = True
                await asyncio.sleep(interval_sec)
                continue
        else:
            shared_state["vstop_active"] = False
            
        if not prices:
            logging.warning("[警告] 価格データの取得に失敗 → スキップ")
            await asyncio.sleep(interval_sec)
            continue

        bid = prices["bid"]
        ask = prices["ask"]
        mid = (ask + bid) / 2

        price_buffer.append(bid)
        high_prices.append(ask)
        low_prices.append(bid)
        close_prices.append(mid)

        logging.info(f"[DEBUG] price_buffer: {len(price_buffer)}")

        if len(price_buffer) < long_period:
            if not shared_state.get("trend_init_notice"):
                notify_slack("[MAトレンド判定] データ蓄積中 → 判定保留中")
                shared_state["trend_init_notice"] = True
            await asyncio.sleep(interval_sec)
            continue

        short_ma = sum(list(price_buffer)[-short_period:]) / short_period
        long_ma = sum(list(price_buffer)[-long_period:]) / long_period
        diff = short_ma - long_ma

        try:
            rsi = calculate_rsi(list(price_buffer), period=14)
            adx = calculate_adx(high_prices, low_prices, close_prices, period=14)
            logging.info(f"[DEBUG] RSI={rsi}, ADX={adx}")
        except Exception as e:
            logging.exception("RSIまたはADXの計算中に例外が発生")
            notify_slack(f"[エラー] RSI/ADX計算中に例外: {e}")
            await asyncio.sleep(interval_sec)
            continue

        if rsi is None or adx is None:
            shared_state["trend"] = None
            logging.warning("[警告] RSIまたはADXがNoneのためスキップ")
            if not shared_state.get("rsi_adx_none_notice", False):
                notify_slack("[注意] RSIまたはADXが未計算のため判定スキップ中")
                shared_state["rsi_adx_none_notice"] = True
            await asyncio.sleep(interval_sec)
            continue
        else:
            shared_state["rsi_adx_none_notice"] = False

        macd, signal = calc_macd(close_prices)
        if len(macd) < 2 or len(signal) < 2:
            notify_slack("[注意] MACDが未計算のため判定スキップ中")
            shared_state["last_skip_notice"] = True
            await asyncio.sleep(interval_sec)
            continue
        else:
            shared_state["last_skip_notice"] = False

        macd_cross_up = macd[-3] <= signal[-3] and macd[-2] <= signal[-2] and macd[-1] > signal[-1]
        macd_cross_down = macd[-3] >= signal[-3] and macd[-2] >= signal[-2] and macd[-1] < signal[-1]

        logging.info(f"[INFO] MACD判定値: UP={macd_cross_up}, DOWN={macd_cross_down}")

        now = datetime.now()
        if now.hour >= 22:
            if not shared_state.get("vstop_active", False):
                notify_slack(f"[クールダウン] 22時以降のため自動売買スキップ")
                shared_state["vstop_active"] = True
                await asyncio.sleep(interval_sec)
                continue
        else:
            shared_state["vstop_active"] = False

        if len(close_prices) >= 5:
            price_range = max(close_prices) - min(close_prices)
            if price_range < 0.03:
                shared_state["trend"] = None
                if not shared_state.get("last_skip_notice", False):
                    notify_slack(f"[横ばい判定] 価格変動幅が小さい（{price_range:.4f}）ためスキップ")
                    shared_state["last_skip_notice"] = True
                await asyncio.sleep(interval_sec)
                continue

        if rsi < 5:
            shared_state["trend"] = None
            if not shared_state.get("last_skip_notice", False):
                notify_slack(f"[RSI下限] RSI={rsi:.2f} → 反発警戒でスキップ")
                shared_state["last_skip_notice"] = True
            await asyncio.sleep(interval_sec)
            continue

        if rsi >= 68:
            rsi_state = "overbought"
        elif rsi <= 32:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"
        shared_state["RSI"] = rsi

        if rsi_state != last_rsi_state:
            if rsi_state == "overbought":
                notify_slack(f"[RSI] 買われすぎ (RSI={rsi:.2f}) → スキップ")
            elif rsi_state == "oversold":
                notify_slack(f"[RSI] 売られすぎ (RSI={rsi:.2f}) → スキップ")
            last_rsi_state = rsi_state

        if adx < 20 and last_adx_state != "weak":
            notify_slack(f"[ADX] トレンドが弱いため抑制中 (ADX={adx:.2f})")
            last_adx_state = "weak"
        elif adx >= 20:
            last_adx_state = "strong"

        trend = None
        if statistics.stdev(list(price_buffer)[-5:]) > VOL_THRESHOLD:
            trend = "BUY" if diff > 0 else "SELL"
            if trend == "BUY" and macd_cross_up:
                shared_state["trend"] = trend
                notify_slack(f"[トレンド] MACDクロスBUY（RSI={rsi:.2f}, ADX={adx:.2f}）")
            elif trend == "SELL" and macd_cross_down:
                shared_state["trend"] = trend
                notify_slack(f"[トレンド] MACDクロスSELL（RSI={rsi:.2f}, ADX={adx:.2f}）")
            else:
                shared_state["trend"] = None
                if not shared_state.get("last_skip_notice", False):
                    notify_slack(f"[スキップ] MACDクロス未検出のためスキップ（RSI={rsi:.2f}, ADX={adx:.2f}）")
                    shared_state["last_skip_notice"] = True
                continue

        await asyncio.sleep(interval_sec)


# === 署名作成 ===
def create_signature(timestamp, method, path, body=""):
    message = timestamp + method + path + body
    return hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()

# === 営業状態チェック ===
def is_market_open():
    try:
        response = requests.get(f"{FOREX_PUBLIC_API}/v1/status")
        response.raise_for_status()
        status = response.json().get("data", {}).get("status")
        # notify_slack(f"[市場] ステータス: {status}")
        return status
    except Exception as e:
        logging.error(f"[市場] 状態取得失敗: {e}")
        return False

# === 建玉取得 ===
def get_positions():
    path = "/v1/openPositions"
    method = "GET"
    timestamp = str(int(time.time() * 1000))
    sign = create_signature(timestamp, method, path)

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign,
    }

    try:
        res = requests.get(BASE_URL_FX + path, headers=headers)
        res.raise_for_status()
        data = res.json().get("data", {})

        positions = data.get("list", [])
        if not isinstance(positions, list):
            logging.warning(f"[建玉] list が見つからない: {data}")
            return []

        return [p for p in positions if p.get("symbol") == SYMBOL]
    except Exception as e:
        logging.error(f"[建玉] 取得失敗: {e}")
        return []

# === 証拠金維持率取得 ===
def get_margin_status(shared_state):
    path = "/v1/account/assets"
    method = "GET"
    timestamp = str(int(time.time() * 1000))
    sign = create_signature(timestamp, method, path)

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    try:
        res = requests.get(BASE_URL_FX + path, headers=headers)
        res.raise_for_status()
        data = res.json().get("data", {})
        ratio_raw = data.get("marginRatio")

        if ratio_raw is None or ratio_raw == 0 or float(ratio_raw) > 1e6:
            if shared_state.get("last_margin_notify") != "none":
                notify_slack("[証拠金維持率] ポジションが存在しないため未算出です。※現在0またはNone")
                shared_state["last_margin_notify"] = "none"
            return

        ratio = float(ratio_raw)

        # 差分が大きい時だけ通知
        last_ratio = shared_state.get("last_margin_ratio")
        if last_ratio is None or abs(ratio - last_ratio) > 1.0:
            notify_slack(f"[証拠金維持率] {ratio:.2f}%")
            shared_state["last_margin_ratio"] = ratio
            shared_state["last_margin_notify"] = "ok"

        # 危険水準通知も重複制御
        if ratio < MAINTENANCE_MARGIN_RATIO * 100:
            if shared_state.get("margin_alert_sent") != True:
                notify_slack("[⚠️アラート] 証拠金維持率が危険水準")
                shared_state["margin_alert_sent"] = True
        else:
            shared_state["margin_alert_sent"] = False

    except Exception as e:
        notify_slack(f"[証拠金] 取得失敗: {e}")

# === 注文発行 ===
def open_order(side="BUY"):
    path = "/v1/order"
    method = "POST"
    timestamp = str(int(time.time() * 1000))  # より正確なミリ秒

    body_dict = {
        "symbol": SYMBOL,
        "side": side,
        "executionType": "MARKET",
        "size": str(LOT_SIZE),
        "symbolType": "FOREX"
    }

    body = json.dumps(body_dict, separators=(',', ':'))
    sign = create_signature(timestamp, method, path, body)

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign,
        "Content-Type": "application/json"
    }

    try:
        start = time.time()
        res = requests.post(BASE_URL_FX + path, headers=headers, data=body)
        end = time.time()
        price = extract_price_from_response(res)
        elapsed = end - start
        data = res.json()

        # 成功・失敗判定と詳細通知
        if res.status_code == 200 and "data" in data:
            #price = data["data"].get("price", "取得不可")
            notify_slack(f"[注文] 新規建て成功: {side}（約定価格: {price}）")
        else:
            notify_slack(f"[注文] 新規建て応答異常: {res.status_code} {data}")

        # 遅延が0.5秒超えたら警告
        if elapsed > 0.5:
            logging.warning(f"[遅延警告] 新規注文に {elapsed:.2f} 秒かかりました")

        return data
    except Exception as e:
        notify_slack(f"[注文] 新規建て失敗: {e}")
        return None


# === ポジション決済 ===
def close_order(position_id, size, side):
    path = "/v1/closeOrder"
    method = "POST"
    timestamp = str(int(time.time() * 1000))  # より精度の高いミリ秒

    body_dict = {
        "symbol": SYMBOL,
        "side": side,
        "executionType": "MARKET",
        "settlePosition": [
            {
                "positionId": position_id,
                "size": str(size)
            }
        ]
    }

    body = json.dumps(body_dict, separators=(',', ':'))
    sign = create_signature(timestamp, method, path, body)

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign,
        "Content-Type": "application/json"
    }

    try:
        start = time.time()
        res = requests.post(BASE_URL_FX + path, headers=headers, data=body)
        end = time.time()
        price = extract_price_from_response(res)
        elapsed = end - start
        data = res.json()

        # 成功応答かチェック
        if res.status_code == 200 and "data" in data:
            # price = data["data"].get("price", "取得不可")
            notify_slack(f"[決済] 成功: {side}（約定価格: {price}）")
        else:
            notify_slack(f"[決済] 応答異常: {res.status_code} {data}")

        # 遅延が長い場合ログ記録
        if elapsed > 0.5:
            logging.warning(f"[遅延警告] 決済APIに {elapsed:.2f} 秒かかりました")

        return data
    except Exception as e:
        notify_slack(f"[決済] 失敗: {e}")
        return None


import time

# == 即時利確監視用タスク ==
async def monitor_quick_profit(shared_state, stop_event, interval_sec=1):
    PROFIT_BUFFER = 5  # 利確ラインに対する安全マージン

    while not stop_event.is_set():
        positions = get_positions()
        prices = get_price()
        if prices is None:
            await asyncio.sleep(interval_sec)
            continue

        ask = prices["ask"]
        bid = prices["bid"]

        for pos in positions:
            entry = float(pos["price"])
            pid = pos["positionId"]
            size_str = int(pos["size"])
            side = pos.get("side", "BUY").upper()
            close_side = "SELL" if side == "BUY" else "BUY"

            profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

            entry_time = shared_state.get("entry_time")
            if entry_time is None:
                continue
            elapsed = time.time() - entry_time

            # 利確ライン（スリッページ考慮）
            short_term_target = 10 + PROFIT_BUFFER
            long_term_target = 30 + PROFIT_BUFFER

            if (elapsed <= 60 and profit >= short_term_target) or (elapsed > 60 and profit >= long_term_target):
                notify_slack(f"[即時利確] 利益が {profit} 円（{elapsed:.1f}秒保持）→ 決済実行")

                start = time.time()
                close_order(pid, size_str, close_side)
                end = time.time()

                record_result(profit, shared_state)
                write_log("QUICK_PROFIT", bid)

                elapsed_api = end - start
                if elapsed_api > 0.5:
                    logging.warning(f"[遅延警告] 利確リクエストに {elapsed_api:.2f} 秒かかりました")

                shared_state["trend"] = None
                shared_state["last_trend"] = None
                shared_state["entry_time"] = time.time()

        await asyncio.sleep(interval_sec)

from threading import Event
trend_none_count = 0
# === メイン処理 ===
stop_event = Event()

# メイン取引処理
async def auto_trade():
    global trend_none_count
    vstop = 0
    hold_status_task = asyncio.create_task(monitor_hold_status(shared_state, stop_event, interval_sec=1))
    trend_task = asyncio.create_task(monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=shared_state))
    trend_task.add_done_callback(lambda t: notify_slack(f"トレンド関数が終了しました: {t.exception()}"))
    loss_cut_task = asyncio.create_task(monitor_positions_fast(shared_state, stop_event, interval_sec=1))
    quit_profit=asyncio.create_task(monitor_quick_profit(shared_state, stop_event))
    quit_profit.add_done_callback(lambda t: notify_slack(f"即時利確関数が終了しました: {t.exception()}"))
    
    
    try:
        while True:
            if is_market_open() != "OPEN":
                if vstop==0:
                    notify_slack(f"[市場] 市場がCLOSEかメンテナンス中")
                    vstop = 1
                continue
            else:
                vstop = 0
            get_margin_status(shared_state)
            positions = get_positions()
            prices = get_price()
            if prices is None:
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            ask = prices["ask"]
            bid = prices["bid"]
            
            spread = abs(ask - bid) 
            last_spread = shared_state.get("last_spread")
            
            if spread > MAX_SPREAD:
                if last_spread is None or abs(spread - last_spread) >= 0.001 :
                    notify_slack(f"[スプレッド] {spread:.3f}円 → スプレッドが広すぎるため見送り")
                    shared_state["last_spread"] = spread
            else:
                shared_state["last_spread"] = None  # 通常状態に戻したい場合

            trend = shared_state.get("trend")
            
            if not positions:
                trend = shared_state.get("trend")
                
                if trend is None:
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue
                else:
                    notify_slack(f"[建玉] なし → 新規{trend}")
                    try:
                        open_order(trend)
                        shared_state["entry_time"] = time.time()
                        write_log(trend, ask)
                    except Exception as e:
                        notify_slack(f"[注文失敗] {e}")

            else:
                for pos in positions:
                    entry = float(pos["price"])
                    pid = pos["positionId"]
                    size_str = int(pos["size"])
                    side = pos.get("side", "BUY").upper()
                    close_side = "SELL" if side == "BUY" else "BUY"
                    spread = abs(prices["ask"] - prices["bid"])
                    spread_history.append(spread)
                    if len(spread_history) == spread_history.maxlen:
                        if all(s > MAX_SPREAD for s in spread_history):
                            close_order(pid, size_str, close_side)
                            write_log("LOSS_CUT", bid)
                            notify_slack("[即時損切] スプレッドが一定時間連続で拡大。ポジションを解消しました。")
                    profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)
                    rsi=shared_state.get("RSI")
                    if profit >= MIN_PROFIT:
                        notify_slack(f"[決済] 利確条件（利益が {profit} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        record_result(profit, shared_state)
                        write_log("SELL", bid)
                        shared_state["trend"]=None
                        shared_state["last_trend"]=None
                        shared_state["entry_time"] = time.time()
                    # RSI反発による利確（BUYのみ例示）
                    elif side == "BUY" and rsi >= 45 and profit > 0:
                        notify_slack(f"[決済] RSI反発による早期利確（RSI: {rsi:.2f}, 利益: {profit:.2f} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        record_result(profit, shared_state)
                        write_log("RSI_PROFIT", bid)
                        shared_state["trend"] = None
                        shared_state["last_trend"] = None
                        shared_state["entry_time"] = time.time()
                    elif profit <= -MAX_LOSS:
                        notify_slack(f"[決済] 損切り条件（損失が {profit} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        record_result(profit, shared_state)
                        write_log("LOSS_CUT", bid)
                        shared_state["trend"]=None
                        shared_state["last_trend"]=None
                        shared_state["entry_time"] = time.time()

                    elif close_side == "SELL" and rsi <= 55 and profit > 0:
                        notify_slack(f"[決済] RSI反落による早期利確（RSI: {rsi:.2f}, 利益: {profit:.2f} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        record_result(profit, shared_state)
                        write_log("RSI_PROFIT", bid)
                        shared_state["trend"] = None
                        shared_state["last_trend"] = None
                        shared_state["entry_time"] = time.time()
            await asyncio.sleep(CHECK_INTERVAL)

    finally:
        stop_event.set()
        trend_task.cancel()
        loss_cut_task.cancel()
        quit_profit.cancel()
        hold_status_task.cancel()
        try:
            await hold_status_task
        except asyncio.CancelledError:
            notify_slack("[INFO] monitor_hold_status タスク終了")
        try:
            await trend_task
        except asyncio.CancelledError:
            notify_slack("[INFO] monitor_trend タスク終了")
        try:
            await loss_cut_task
        except asyncio.CancelledError:
            notify_slack("[INFO] monitor_positions_fast タスク終了")
        try:
            await quit_profit
        except asyncio.CancelledError:
            notify_slack("[INFO] monitor_quick_profit タスク終了")
if __name__ == "__main__":
    try:
        asyncio.run(auto_trade())
    except:
        save_state(shared_state)
        save_price_buffer(price_buffer)