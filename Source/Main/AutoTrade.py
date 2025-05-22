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
import asyncio
import statistics
import pandas as pd
import statistics
import signal
from collections import deque
import mysql.connector
from state_utils import (
    save_state,
    load_state,
    save_price_buffer,
    load_price_buffer
)

notify_slack("自動売買システム起動")
shared_state = load_state()
price_buffer = load_price_buffer()

LOG_FILE = "fx_trade_log.csv"

DEFAULT_CONFIG = {
    "LOT_SIZE": 1000,
    "MAX_SPREAD": 0.03,
    "MAX_LOSS": 20,
    "MIN_PROFIT": 40,
    "CHECK_INTERVAL": 3,
    "MAINTENANCE_MARGIN_RATIO": 0.5,
    "VOL_THRESHOLD": 0.03
}

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
# == 損益即時監視用タスク==
async def monitor_positions_fast(shared_state, stop_event, interval_sec=1):
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

            if profit <= -MAX_LOSS:
                notify_slack(f"[即時損切] 損失が {profit} 円 → 強制決済実行")
                close_order(pid, size_str, close_side)
                write_log("LOSS_CUT_FAST", bid)

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
    "rsi_adx_none_notice":False
}

# === 環境変数の読み込み ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL_FX = "https://forex-api.coin.z.com/private"
FOREX_PUBLIC_API = "https://forex-api.coin.z.com/public"

# === トレンド判定関数 ===
signal.signal(signal.SIGTERM, handle_exit)
# monitor_trend() の外で共有してもOK（必要に応じて）

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
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()

    return adx.iloc[-1]

# === トレンド判定を拡張（RSI+ADX込み） ===
async def monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=None):
    last_rsi_state = None  # rsiの状態を追跡
    last_adx_state = None  # adx状態の変化通知

    high_prices = []
    low_prices = []
    close_prices = []

    while not stop_event.is_set():
        p = get_price()
        if p:
            price_buffer.append(p["bid"])
            high_prices.append(p["ask"])
            low_prices.append(p["bid"])
            close_prices.append((p["ask"] + p["bid"]) / 2)

            # 保持長さ制限（最大240）
            if len(high_prices) > 240:
                high_prices.pop(0)
                low_prices.pop(0)
                close_prices.pop(0)

        if len(price_buffer) < long_period:
            if not shared_state.get("trend_init_notice"):
                notify_slack("[MAトレンド判定] データ蓄積中 → 判定保留中")
                shared_state["trend_init_notice"] = True
            await asyncio.sleep(interval_sec)
            continue

        short_ma = sum(list(price_buffer)[-short_period:]) / short_period
        long_ma = sum(list(price_buffer)[-long_period:]) / long_period
        prev_short = shared_state.get("last_short_ma")
        prev_long = shared_state.get("last_long_ma")

        short_ma_diff = abs(short_ma - prev_short) if prev_short is not None else 999
        long_ma_diff = abs(long_ma - prev_long) if prev_long is not None else 999

        if short_ma_diff > 0.03 or long_ma_diff > 0.03:
            shared_state["last_short_ma"] = short_ma
            shared_state["last_long_ma"] = long_ma

        diff = short_ma - long_ma
        rsi = calculate_rsi(list(price_buffer), period=14)
        adx = calculate_adx(high_prices, low_prices, close_prices, period=14)
        
        # --- RSI / ADX が未計算の場合はスキップ ---
        if rsi is None or adx is None:
            shared_state["trend"] = None
            if not shared_state.get("rsi_adx_none_notice", False):
                notify_slack("[注意] RSIまたはADXが未計算のため判定スキップ中")
                shared_state["rsi_adx_none_notice"] = True
            await asyncio.sleep(interval_sec)
            continue
        else:
            shared_state["rsi_adx_none_notice"] = False

        rsi_state = None
        if rsi is not None:
            if rsi >= 70:
                rsi_state = "overbought"
            elif rsi <= 30:
                rsi_state = "oversold"
            else:
                rsi_state = "neutral"

        # RSIの状態に変化があった場合のみ通知
        if rsi_state != last_rsi_state:
            if rsi_state == "overbought":
                shared_state["trend"] = None
                notify_slack(f"[RSI] 買われすぎ (RSI={rsi:.2f}) → スキップ")
            elif rsi_state == "oversold":
                shared_state["trend"] = None
                notify_slack(f"[RSI] 売られすぎ (RSI={rsi:.2f}) → スキップ")
            last_rsi_state = rsi_state

        # ADXによるトレンド制限の通知（変化があったときのみ）
        if adx is not None and adx < 20:
            if last_adx_state != "weak":
                notify_slack(f"[ADX] トレンドが弱いため抑制中 (ADX={adx:.2f})")
                last_adx_state = "weak"
        elif adx is not None and adx >= 20:
            last_adx_state = "strong"

        # MAトレンドとRSIとADXによる判断
        if abs(diff) >= 0.03 and rsi_state == "neutral" and (adx is None or adx >= 20):
            trend = "BUY" if diff > 0 else "SELL"
            shared_state["trend"] = trend
            shared_state["last_skip_notice"] = False
            if shared_state.get("last_trend") != trend:
                notify_slack(f"[MA+RSI+ADXトレンド] → トレンド方向は {trend} (RSI={rsi:.2f}, ADX={adx:.2f})")
                shared_state["last_trend"] = trend

        elif len(price_buffer) >= 5 and statistics.stdev(list(price_buffer)[-5:]) > VOL_THRESHOLD:
            if rsi_state == "neutral" and (adx is None or adx >= 20):
                trend = "BUY" if diff > 0 else "SELL"
                shared_state["trend"] = trend
                shared_state["last_skip_notice"] = False
                notify_slack(f"[ボラティリティ判定] 差小だが高ボラ → 強制トレンド方向は {trend}")
                shared_state["last_trend"] = trend
            else:
                shared_state["trend"] = None
                if not shared_state.get("last_skip_notice", False):
                    notify_slack(f"[ボラティリティ判定] 高ボラだがRSI/ADX条件満たさず → スキップ (RSI={rsi:.2f}, ADX={adx:.2f})")
                    shared_state["last_skip_notice"] = True

        else:
            shared_state["trend"] = None
            if not shared_state.get("last_skip_notice", False):
                notify_slack("[MAトレンド判定] 差が小さく方向不明 → スキップ")
                shared_state["last_skip_notice"] = True

        await asyncio.sleep(interval_sec)


# === ログ設定 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("fx_debug_log.txt"),
        logging.StreamHandler()
    ]
)

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
        notify_slack(f"[市場] ステータス: {status}")
        return status == "OPEN"
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

        if ratio_raw is None or ratio_raw == 0:
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
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
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
        res = requests.post(BASE_URL_FX + path, headers=headers, data=body)
        notify_slack(f"[注文] 新規建て: {side}")
        return res.json()
    except Exception as e:
        notify_slack(f"[注文] 新規建て失敗: {e}")
        return None

# === ポジション決済 ===
def close_order(position_id, size,side):
    path = "/v1/closeOrder"
    method = "POST"
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    body_dict = {
        "symbol": SYMBOL,
        "side": side,
        "executionType": "MARKET",
        "settlePosition": [
            {
                "positionId": position_id,
                "size": str(size)  # 通貨単位
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
        res = requests.post(BASE_URL_FX + path, headers=headers, data=body)
        notify_slack(f"[決済] 成功: {side}")
        return res.json()
    except Exception as e:
        notify_slack(f"[決済] 失敗: {e}")
        return None

# === ログ記録 ===
def write_log(action, price):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "action", "price"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, price])

from threading import Event
trend_none_count = 0
# === メイン処理 ===
stop_event = Event()

# メイン取引処理
async def auto_trade():
    global trend_none_count
    
    trend_task = asyncio.create_task(monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=shared_state))
    loss_cut_task = asyncio.create_task(monitor_positions_fast(shared_state, stop_event, interval_sec=1))
    if not is_market_open():
        pass
    try:
        while True:
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

                    profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

                    if profit >= MIN_PROFIT:
                        notify_slack(f"[決済] 利確条件（利益が {profit} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        write_log("SELL", bid)
                        trend=None
                    elif profit <= -MAX_LOSS:
                        notify_slack(f"[決済] 損切り条件（損失が {profit} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        write_log("LOSS_CUT", bid)
                        trend=None
                    else:
                        if abs(profit) > 10:
                            notify_slack(f"[保有] 継続 {profit}円")

            await asyncio.sleep(CHECK_INTERVAL)

    finally:
        stop_event.set()
        trend_task.cancel()
        loss_cut_task.cancel()
        try:
            await trend_task
        except asyncio.CancelledError:
            notify_slack("[INFO] monitor_trend タスク終了")
        try:
            await loss_cut_task
        except asyncio.CancelledError:
            notify_slack("[INFO] monitor_positions_fast タスク終了")
            
if __name__ == "__main__":
    try:
        asyncio.run(auto_trade())
    except:
        save_state(shared_state)
        save_price_buffer(price_buffer)