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
from state_utils import (
    save_state,
    load_state,
    save_price_buffer,
    load_price_buffer,
    save_adx_buffers,
    load_adx_buffers
)

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
    "entry_time":None
}

TEST = False # デバッグ用フラグ
if os.path.exists("fx_debug_log.txt")==True:
    os.remove("fx_debug_log.txt")

# ===ログ設定 ===
LOG_FILE1 = "fx_debug_log.txt"
_log_last_reset = datetime.now()
def setup_logging():
    """初期ログ設定（起動時）"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE1, mode='a', encoding='utf-8'),
        ]
    )
    
try:
    setup_logging()
except Exception as e:
    print(f"ログ初期化時にエラー: {e}")
notify_slack("自動売買システム起動")

# == 記録済みデータ読み込み ===
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
                shared_state["trend"] = None
                shared_state["last_trend"] = None
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

# 1時間ごとにログファイルを初期化（TEST時はスキップ)
def reset_logging_if_needed():
    
    global _log_last_reset
    if TEST:
        return

    now = datetime.now()
    if now - _log_last_reset >= timedelta(hours=1):
        _log_last_reset = now

        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        # ファイルを初期化（中身を消す）
        open(LOG_FILE1, "w").close()

        # 再設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE1, mode='a', encoding='utf-8'),
            ]
        )
        logging.info("[INFO] ログを初期化しました（1時間ごと）")

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

# === トレンド判定を拡張（RSI+ADX込み） ===
async def monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=None):
    last_rsi_state = None
    last_adx_state = None

    high_prices = deque(maxlen=240)
    low_prices = deque(maxlen=240)
    close_prices = deque(maxlen=240)

    while not stop_event.is_set():
        reset_logging_if_needed()
        p = get_price()
        if not p:
            logging.warning("[警告] 価格データの取得に失敗 → スキップ")
            await asyncio.sleep(interval_sec)
            continue

        logging.info(f"[DEBUG] price_buffer: {len(price_buffer)}")

        price_buffer.append(p["bid"])
        high_prices.append(p["ask"])
        low_prices.append(p["bid"])
        close_prices.append((p["ask"] + p["bid"]) / 2)

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


        if shared_state.get("entry_time"):
            elapsed = datetime.now() - shared_state["entry_time"]
            if elapsed.total_seconds() < 60:
                shared_state["trend"] = None
                shared_state["last_trend"] = None
                notify_slack(f"[クールダウン] 前回決済から{elapsed.total_seconds():.1f}秒 → スキップ")
                shared_state["last_skip_notice"] = True
                continue

        if rsi < 5:
            shared_state["trend"] = None
            if not shared_state.get("last_skip_notice", False):
                notify_slack(f"[RSI下限] RSI={rsi:.2f} → 反発警戒でスキップ")
                shared_state["last_skip_notice"] = True
                continue

        if rsi >= 68:
            rsi_state = "overbought"
        elif rsi <= 32:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"
        shared_state["RSI"] = rsi

        prices = get_price()
        if not prices:
            logging.warning("[警告] トレンド判定中に価格取得失敗 → スキップ")
            await asyncio.sleep(interval_sec)
            continue
        ask = prices["ask"]
        bid = prices["bid"]
        spread = abs(ask - bid)

        logging.info(f"[DEBUG] MA差: diff={diff}, spread={spread}")
        logging.info(f"[DEBUG] RSI状態: {rsi_state}")

        if rsi_state != last_rsi_state:
            if rsi_state == "overbought":
                shared_state["trend"] = None
                notify_slack(f"[RSI] 買われすぎ (RSI={rsi:.2f}) → スキップ")
            elif rsi_state == "oversold":
                shared_state["trend"] = None
                notify_slack(f"[RSI] 売られすぎ (RSI={rsi:.2f}) → スキップ")
            last_rsi_state = rsi_state

        if adx < 20 and last_adx_state != "weak":
            notify_slack(f"[ADX] トレンドが弱いため抑制中 (ADX={adx:.2f})")
            last_adx_state = "weak"
        elif adx >= 20:
            last_adx_state = "strong"

        if len(price_buffer) >= 5 and statistics.stdev(list(price_buffer)[-5:]) > VOL_THRESHOLD:
            trend = "BUY" if diff > 0 else "SELL"

            if rsi_state == "neutral" and adx >= 25:
                prices = get_price()
                if prices:
                    spread = abs(prices["ask"] - prices["bid"])
                else:
                    logging.warning("[警告] 再取得価格がNone → spread保持")
                
            if rsi < 15 or rsi > 85:
                shared_state["trend"] = None
                if not shared_state.get("last_skip_notice", False):
                    notify_slack(f"[ボラティリティ判定] RSI過熱のためエントリースキップ (RSI={rsi:.2f}, ADX={adx:.2f})")
                    shared_state["last_skip_notice"] = True
                else:
                    shared_state["last_skip_notice"] = False
            elif spread <= MAX_SPREAD:
                if adx > 20:
                    shared_state["trend"] = trend
                    shared_state["last_skip_notice"] = False
                    shared_state["last_trend"] = trend
                    notify_slack(f"[ボラティリティ判定] 高ボラ強制トレンド方向は {trend}（RSI={rsi:.2f}, ADX={adx:.2f}）")
                else:
                    shared_state["trend"] = trend
                    shared_state["last_skip_notice"] = False
                    shared_state["last_trend"] = trend
                    notify_slack(f"[スキップ] ADX={adx:.2f} でトレンド不明確のためトレード抑制")
            else:
                shared_state["trend"] = None
                if not shared_state.get("last_skip_notice", False):
                    notify_slack(f"[ボラティリティ判定] 高ボラだがスプレッドが広いためスキップ (RSI={rsi:.2f}, ADX={adx:.2f})")
                    shared_state["last_skip_notice"] = True
        else:
            shared_state["trend"] = None
            if shared_state.get("last_trend") and shared_state["last_trend"] != trend:
                notify_slack(f"[建玉スキップ] 高ボラ中に方向反転検知（{shared_state['last_trend']}→{trend}）")
                shared_state["last_skip_notice"] = True
            elif not shared_state.get("last_skip_notice", False):
                notify_slack(f"[ボラティリティ判定] 高ボラだがRSI/ADX条件満たさず → スキップ (RSI={rsi:.2f}, ADX={adx:.2f})")
                shared_state["last_skip_notice"] = True
            else:
                shared_state["trend"] = None
                if not shared_state.get("last_skip_notice", False):
                    notify_slack("[MAトレンド判定] 差が小さく方向不明 → スキップ")
                    shared_state["last_skip_notice"] = True

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
        notify_slack(f"[市場] ステータス: {status}")
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

# === 取引ログ記録 ===
def write_log(action, price):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "action", "price"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, price])

import time

async def monitor_quick_profit(shared_state, stop_event, interval_sec=3):
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

            # 利益計算
            profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

            # entry_timeが記録されている前提
            entry_time = shared_state.get("entry_time", 0)
            elapsed = time.time() - entry_time

            # 即時利確条件: 60秒以内 & 利益10円以上
            if profit >= 10 and elapsed <= 60:
                notify_slack(f"[即時利確] 利益が {profit} 円（{elapsed:.1f}秒保持）→ 決済実行")
                close_order(pid, size_str, close_side)
                write_log("QUICK_PROFIT", bid)
                shared_state["trend"] = None
                shared_state["last_trend"] = None

        await asyncio.sleep(interval_sec)

from threading import Event
trend_none_count = 0
# === メイン処理 ===
stop_event = Event()

# メイン取引処理
async def auto_trade():
    global trend_none_count

    hold_status_task = asyncio.create_task(monitor_hold_status(shared_state, stop_event, interval_sec=1))
    trend_task = asyncio.create_task(monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=shared_state))
    loss_cut_task = asyncio.create_task(monitor_positions_fast(shared_state, stop_event, interval_sec=1))
    quit_profit=asyncio.create_task(monitor_quick_profit(shared_state, stop_event))
    
    if is_market_open() != "OPEN":
        notify_slack(f"[市場] 市場がCLOSEかメンテナンス中")
        sys.exit(0)
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

                    profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)
                    rsi=shared_state.get("RSI")
                    if profit >= MIN_PROFIT:
                        notify_slack(f"[決済] 利確条件（利益が {profit} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        write_log("SELL", bid)
                        shared_state["trend"]=None
                        shared_state["last_trend"]=None
                    # RSI反発による利確（BUYのみ例示）
                    elif side == "BUY" and rsi >= 45 and profit > 0:
                        notify_slack(f"[決済] RSI反発による早期利確（RSI: {rsi:.2f}, 利益: {profit:.2f} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        write_log("RSI_PROFIT", bid)
                        shared_state["trend"] = None
                        shared_state["last_trend"] = None
                    elif profit <= -MAX_LOSS:
                        notify_slack(f"[決済] 損切り条件（損失が {profit} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        write_log("LOSS_CUT", bid)
                        shared_state["trend"]=None
                        shared_state["last_trend"]=None
                    elif close_side == "SELL" and rsi <= 55 and profit > 0:
                        notify_slack(f"[決済] RSI反落による早期利確（RSI: {rsi:.2f}, 利益: {profit:.2f} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        write_log("RSI_PROFIT", bid)
                        shared_state["trend"] = None
                        shared_state["last_trend"] = None

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