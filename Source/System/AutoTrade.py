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
from socket_server import start_socket_server
from state_utils import (
    save_state,
    load_state,
    save_price_buffer,
    load_price_buffer,
    load_price_history,
    save_price_history
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
    "vstop_active":False,
    "adx_wait_notice":False,
    "forced_entry_date":False,
    "cmd":None,
    "trend_start_time":None,
    "oders_error":False
}

import configparser
def load_ini():
    # ConfigParser オブジェクトを作成
    config = configparser.ConfigParser()

    # config.ini を読み込む
    config.read('config.ini')
    reset = config.getboolean('settings', 'reset')
    return reset

reset = load_ini()
args=sys.argv
file_path = sys.argv[0]  # スクリプトファイルのパス
folder_path = os.path.dirname(os.path.abspath(file_path))
os.chdir(folder_path)

session = requests.Session() # セッションを生成

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

try:
    setup_logging()
except Exception as e:
    print(f"ログ初期化時にエラー: {e}")
notify_slack("自動売買システム起動")

# == 記録済みデータ読み込み ===
shared_state = load_state()
price_buffer = load_price_buffer()

# LOG_FILE = "fx_trade_log.csv"
LOSS_STREAK_THRESHOLD = 3
COOLDOWN_DURATION_SEC = 180  # 3分間

DEFAULT_CONFIG = {
    "LOT_SIZE": 1000,
    "MAX_SPREAD": 0.03,
    "MAX_LOSS": 20,
    "MIN_PROFIT": 40,
    "CHECK_INTERVAL": 3,
    "MAINTENANCE_MARGIN_RATIO": 0.5,
    "VOL_THRESHOLD": 0.03,
    "TIME_STOP":22,
    "MACD_DIFF_THRESHOLD":0.002,
    "SKIP_MODE":0,
    "SYMBOL":"USD_JPY"
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
                original_type = type(DEFAULT_CONFIG[key])
                try:
                    if original_type == int:
                        config[key] = int(value)
                    elif original_type == float:
                        config[key] = float(value)
                    elif original_type == str:
                        config[key] = str(value)
                    elif original_type == bool:
                        config[key] = value.lower() in ['true', '1', 'yes']
                except Exception:
                    pass  # 型変換に失敗してもスキップ
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
            bid = prices["bid"]
            ask = prices["ask"]
            #mid = (ask + bid) / 2

            spread = ask - bid
            
            if profit <= (-MAX_LOSS + SLIPPAGE_BUFFER):
                if spread > MAX_SPREAD:
                    notify_slack(f"[即時損切保留] 強制決済実行の条件に達したが、スプレッドが拡大中なのでスキップ\n 損切タイミングに注意")
                    continue
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
SYMBOL = config["SYMBOL"]
LOT_SIZE = config["LOT_SIZE"]
MAX_SPREAD = config["MAX_SPREAD"]
MAX_LOSS = config["MAX_LOSS"]
MIN_PROFIT = config["MIN_PROFIT"]
CHECK_INTERVAL = config["CHECK_INTERVAL"]
MAINTENANCE_MARGIN_RATIO = config["MAINTENANCE_MARGIN_RATIO"]
VOL_THRESHOLD = config["VOL_THRESHOLD"]
TIME_STOP = config["TIME_STOP"]
MACD_DIFF_THRESHOLD =config["MACD_DIFF_THRESHOLD"]
SKIP_MODE = config["SKIP_MODE"] # 差分が小さい場合にスキップするかどうか、スキップする場合はTrue

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
    if len(highs) < period + 2:
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

def fee_test(trend):
    """ 
    手数料から約定金額を算出するコード
    trend: "BUY" または "SELL"
    """
    price_data = get_price()
    if not price_data:
        logging.error("価格データが取得できませんでした")
        return
    if trend == "BUY":
        price = price_data["ask"]  # 買い注文は ask で約定
    elif trend == "SELL":
        price = price_data["bid"]  # 売り注文は bid で約定
    else:
        logging.error(f"無効なトレンド指定: {trend}")
        return
    amount = 0.1 * 10000 * price  # 0.1lot = 1000通貨、1lot = 10000通貨
    fee = amount * 0.00002  # 0.002%
    notify_slack(f"想定手数料は、{fee:.3f} 円です")
    logging.info(f"想定手数料: {fee:.3f} 円 (ロット: {LOT_SIZE}, レート: {price}, 約定金額: {amount:.2f})")
        
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
        res = session.post(BASE_URL_FX + path, headers=headers, data=body,timeout=3)
        end = time.time()
        price = extract_price_from_response(res)
        elapsed = end - start
        data = res.json()

        # 成功・失敗判定と詳細通知
        if res.status_code == 200 and "data" in data:
            #price = data["data"].get("price", "取得不可")
            notify_slack(f"[注文] 新規建て成功 {side}")
            fee_test(side)
            shared_state["oders_error"]=False
        else:
            notify_slack(f"[注文] 新規建て応答異常: {res.status_code} {data}")
            shared_state["oders_error"]=True
        # 遅延が0.5秒超えたら警告
        if elapsed > 0.5:
            logging.warning(f"[遅延警告] 新規注文に {elapsed:.2f} 秒かかりました")

        return data
    except requests.exceptions.Timeout:
        notify_slack("[注文] タイムアウト（3秒）")
        logging.warning("[タイムアウト] 新規注文が3秒を超えました")
        return None
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
        res = session.post(BASE_URL_FX + path, headers=headers, data=body,timeout=3)
        end = time.time()
        price = extract_price_from_response(res)
        elapsed = end - start
        data = res.json()

        # 成功応答かチェック
        if res.status_code == 200 and "data" in data:
            # price = data["data"].get("price", "取得不可")
            notify_slack(f"[決済] 成功: {side}")
            fee_test(side)
            shared_state["oders_error"]=False
        else:
            notify_slack(f"[決済] 応答異常: {res.status_code} {data}")
            shared_state["oders_error"]=True
        # 遅延が長い場合ログ記録
        if elapsed > 0.5:
            logging.warning(f"[遅延警告] 決済APIに {elapsed:.2f} 秒かかりました")

        return data
    except requests.exceptions.Timeout:
        notify_slack("[注文] タイムアウト（3秒）")
        logging.warning("[タイムアウト] 新規注文が3秒を超えました")
        return None
    except Exception as e:
        notify_slack(f"[決済] 失敗: {e}")
        return None

def first_oder(trend,shared_state=None):
    positions = get_positions()
    prices = get_price()
    if prices is None:
        return 0
    
    bid = prices["bid"]
    ask = prices["ask"]
    spread = ask - bid
    if spread > MAX_SPREAD:
        notify_slack(f"[警告] スプレッドに差が許容範囲外なので取引中止")
        return 3
    if not positions:
        if trend is None:
           return 0
        else:
            notify_slack(f"[建玉] なし → 新規{trend}")
            try:
                open_order(trend)
                shared_state["entry_time"] = time.time()
                write_log(trend, ask)
                return 1
            except Exception as e:
                notify_slack(f"[注文失敗] {e}")
                return 0
    else:
        return 2

# === トレンド判定を拡張（RSI+ADX込み） ===
async def monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=None):
    import statistics
    from collections import deque
    from datetime import datetime
    from datetime import date
    import time
    import logging

    global price_buffer
    # price_buffer = deque(maxlen=240)

    high_prices, low_prices, close_prices = load_price_history()
    xstop = 0
    try:
        trend = shared_state["trend"]
    except:
        trend = None

    last_rsi_state = None
    last_adx_state = None
    sstop = 0
    vstop = 0
    nstop = 0
    timestop = 0

    while not stop_event.is_set():
        positions = get_positions()    
        if shared_state.get("cmd") == "save_adx":
            if len(high_prices) > 28 or len(low_prices) > 28 or len(close_prices) > 28:
                save_price_history(list(high_prices), list(low_prices), list(close_prices))
                notify_slack("[保存] 外部コマンドによりADX蓄積データを保存しました")
                shared_state["cmd"] = None  # フラグをリセット
            else:
                notify_slack("[保存スキップ] 外部コマンドによりADX蓄積データを要求されましたが、データ不足です")
                shared_state["cmd"] = None

        today = datetime.now()
        weekday_number = today.weekday()
        if is_market_open() != "OPEN" or weekday_number == 6 or weekday_number == 5:
            if sstop == 0:
                notify_slack(f"[市場] 市場がCLOSEかメンテナンス中")
                logging.info("[市場] 市場が閉場中")
                sstop = 1
            await asyncio.sleep(interval_sec)
            if weekday_number == 6:
                high_prices.clear()
                low_prices.clear()
                close_prices.clear()
                price_buffer.clear()
                shared_state["price_reset_done"] = True
            continue
        sstop = 0
        in_cd, remaining = is_in_cooldown(shared_state)
        if in_cd:
            if not shared_state.get("notified_cooldown", False):
                notify_slack(f"[クールダウン中] あと{remaining}秒 → エントリー判断を停止中")
                logging.info(f"[クールダウン] 残り{remaining}秒")
                shared_state["notified_cooldown"] = True
            await asyncio.sleep(interval_sec)
            continue
        else:
            shared_state["notified_cooldown"] = False

        prices = get_price()
        now = datetime.now()
        if now.hour < 4:
            high_prices.clear()
            low_prices.clear()
            close_prices.clear()
            price_buffer.clear()
            shared_state["price_reset_done"] = True            
        if now.hour >= TIME_STOP or now.hour < 5:
            if not shared_state.get("vstop_active", False):                   
                notify_slack(f"[クールダウン] {str(TIME_STOP)}時以降のため自動売買スキップ")
                logging.info(f"[時間制限] {str(TIME_STOP)}時以降の取引スキップ")
                shared_state["vstop_active"] = True
                shared_state["forced_entry_date"] = False
                if len(high_prices) < 28 or len(low_prices) < 28 or len(close_prices) < 28:
                    pass
                else:
                    save_price_history(high_prices, low_prices, close_prices)
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

        spread = ask - bid
        if spread > MAX_SPREAD:
            shared_state["trend"] = None
            if nstop== 0:
                notify_slack(f"[スプレッド超過] 現在のスプレッド={spread:.5f} → エントリーをスキップ")
                logging.warning(f"[スキップ] スプレッドが広すぎるため判定中止（{spread:.5f} > {MAX_SPREAD:.5f}）")
                nstop = 1
            continue
        else:
            nstop = 0
            
        if positions:
            bid = prices["bid"]
            ask = prices["ask"]
            #mid = (ask + bid) / 2

            spread = ask - bid
            if spread > MAX_SPREAD:
                shared_state["trend"] = None
                if nstop== 0:
                    notify_slack(f"[スプレッド超過] 現在のスプレッド={spread:.5f} → 取引中にスプレッド拡大\n損切タイミングなどに影響の可能性あり")
                    logging.warning(f"[スキップ] 取引中にスプレッド拡大損切タイミングなどに影響の可能性あり（{spread:.5f} > {MAX_SPREAD:.5f}）")
                    nstop = 1
                continue
            else:
                nstop = 0

        price_buffer.append(bid)
        high_prices.append(ask)
        low_prices.append(bid)
        close_prices.append(mid)
        
        if len(high_prices) < 28 or len(low_prices) < 28 or len(close_prices) < 28:
            logging.info(f"[待機中] ADX計算用に蓄積中: {len(close_prices)}/28")
            await asyncio.sleep(interval_sec)
            continue
        
        if len(price_buffer) < long_period:
            if not shared_state.get("trend_init_notice"):
                notify_slack("[MAトレンド判定] データ蓄積中 → 判定保留中")
                logging.info("[初期化] データ蓄積中")
                shared_state["trend_init_notice"] = True
            await asyncio.sleep(interval_sec)
            continue

        short_ma = sum(list(price_buffer)[-short_period:]) / short_period
        long_ma = sum(list(price_buffer)[-long_period:]) / long_period
        
        sma_cross_up = short_ma > long_ma and shared_state.get("last_short_ma", 0) <= shared_state.get("last_long_ma", 0)
        sma_cross_down = short_ma < long_ma and shared_state.get("last_short_ma", 0) >= shared_state.get("last_long_ma", 0)
        
        logging.info(f"[INFO] SMA クロス SMA_UP = {sma_cross_up} SMA_DOWN = {sma_cross_down}")
        shared_state["last_short_ma"] = short_ma
        shared_state["last_long_ma"] = long_ma
        
        diff = short_ma - long_ma
        
        try:
            rsi = calculate_rsi(list(price_buffer), period=14)
            adx = calculate_adx(high_prices, low_prices, close_prices, period=14)
            rsi_str = f"{rsi:.2f}" if rsi is not None else "None"
            adx_str = f"{adx:.2f}" if adx is not None else "None"
            logging.info(f"[指標] RSI={rsi_str}, ADX={adx_str}")
        except Exception as e:
            rsi_str = str(rsi) if 'rsi' in locals() else "未定義"
            adx_str = str(adx) if 'adx' in locals() else "未定義"
            notify_slack(f"[エラー] RSI/ADX計算中に例外: {e}（RSI={rsi_str}, ADX={adx_str}）")
            logging.exception("RSI/ADX計算中に例外が発生")
            await asyncio.sleep(interval_sec)
            continue

        if rsi is None or adx is None:
            if vstop==0:
               shared_state["trend"] = None
               notify_slack("[注意] RSIまたはADXが未計算のため判定スキップ中")
               logging.warning("[スキップ] RSI/ADXがNone")
               vstop = 1
               await asyncio.sleep(interval_sec)
               continue
        else:
            shared_state["RSI"] = rsi
            vstop = 0
        
        if len(close_prices) < 14:
            logging.info(f"[情報] ADX計算に必要なデータ不足 ({len(close_prices)}/14)")
            if not shared_state.get("adx_wait_notice", False):
                notify_slack("[待機中] ADX計算に必要なデータが不足 → 判定スキップ中")
                shared_state["adx_wait_notice"] = True
                await asyncio.sleep(interval_sec)
            continue
        else:
            shared_state["adx_wait_notice"] = False
            
        macd, signal = calc_macd(close_prices)
        if len(macd) < 2 or len(signal) < 2:
            notify_slack("[注意] MACDが未計算のため判定スキップ中")
            logging.warning("[スキップ] MACD未計算")
            await asyncio.sleep(interval_sec)
            continue

        macd_cross_up = macd[-2] <= signal[-2] and macd[-1] > signal[-1]
        macd_cross_down = macd[-2] >= signal[-2] and macd[-1] < signal[-1]

        macd_bullish = macd[-1] > signal[-1]  # クロスしてる or 継続中    
        macd_bearish = macd[-1] < signal[-1]  # デッドクロスまたは継続中
        
        macd_str = f"{macd[-1]:.5f}" if macd[-1] is not None else "None"
        signal_str = f"{signal[-1]:.5f}" if signal[-1] is not None else "None"
        rsi_limit = (trend == "BUY" and rsi < 70) or (trend == "SELL" and rsi > 30)
        logging.info(f"[MACD] クロス判定: UP={macd_cross_up}, DOWN={macd_cross_down}")
        logging.info(f"[判定詳細] trend候補={trend}, diff={diff:.5f}, stdev={statistics.stdev(list(price_buffer)[-5:]):.5f}")
        if len(close_prices) >= 5:
            price_range = max(close_prices) - min(close_prices)
            if price_range < 0.03:
                trend = None
                shared_state["trend"] = None
                notify_slack(f"[横ばい判定] 価格変動幅が小さい（{price_range:.4f}）ためスキップ")
                logging.info("[スキップ] 価格横ばい")
                await asyncio.sleep(interval_sec)
                continue
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        if adx >= 95:
            # 無効化（非常事態）
            shared_state["trend"] = None
            notify_slack(f"[警告] ADXが100に近いためスキップ（ADX={adx:.2f}）")
            logging.warning("[スキップ] ADX異常値 → 判定中止")
            continue
        elif adx >= 70 and abs(diff) > 0.015 and trend is not None:
            last_forced_entry_date = shared_state.get("forced_entry_date")

            if last_forced_entry_date == today_str:
                logging.info("[強制エントリー制限] 本日すでに実行済みのためスキップ")
                
            else:
                now = datetime.now()
                if now.hour <= 21:
                    timestop = 1
                    # クロス不要で許可
                    shared_state["trend"] = trend
                    try:
                        notify_slack(f"[強トレンド] MACDクロス無視してエントリー（ADX={adx:.2f}, diff={diff:.4f}）")
                    except:
                        pass
                    notify_slack(f"[トレンド] MACDクロス{trend}（RSI={rsi_str}, ADX={adx_str}）")
                    a=first_oder(trend,shared_state)
                    if a==2:
                        logging.info(f"[結果] {trend} すでにポジションあり")
                    elif a==1:
                        logging.info(f"[結果] {trend} 成功")
                        shared_state["oders_error"]=False
                    else:
                        logging.error(f"[結果] {trend} 失敗")
                    if shared_state["oders_error"]==False and a==1:
                        logging.info("[エントリー] ADX強すぎるためクロス無視")
                        shared_state["forced_entry_date"] = today_str
                else:
                    if timestop == 0:
                        notify_slack(f"[情報] MACDクロス無視してエントリーだが、9時以降なのでスキップ")
                        logging.info("[情報] MACDクロス無視してエントリーだが、9時以降なのでスキップ")
                        timestop = 1

        if rsi < 20:
            shared_state["trend"] = None
            notify_slack(f"[RSI下限] RSI={rsi_str} → 反発警戒でスキップ")
            logging.info("[スキップ] RSI下限で警戒")
            await asyncio.sleep(interval_sec)
            continue
                      
        if statistics.stdev(list(price_buffer)[-5:]) > VOL_THRESHOLD:
            
            trend = "BUY" if diff > 0 else "SELL"
            if adx < 20:
                notify_slack(f"[スキップ] ADXが低いためトレンド弱くスキップ（ADX={adx:.2f}）")
                shared_state["trend"] = None
                await asyncio.sleep(interval_sec)
                continue
            
            TREND_HOLD_MINUTES = 15  # 任意の継続時間

            now = datetime.now()
            trend_active = False

            if "trend_start_time" in shared_state:
                elapsed = (now - shared_state["trend_start_time"]).total_seconds() / 60.0
                if elapsed < TREND_HOLD_MINUTES:
                    trend_active = True
                    logging.info(f"[継続中] {shared_state['trend']}トレンド継続中 ({elapsed:.1f}分経過)")

            if trend == "BUY" and (macd_bullish or macd_cross_up) and sma_cross_up and rsi < 70 and adx >= 20:
                shared_state["trend"] = trend
                shared_state["trend_start_time"] = datetime.now()
                notify_slack(f"[トレンド] MACDクロスBUY（RSI={rsi_str}, ADX={adx_str}）")
                a=first_oder(trend,shared_state)
                if a==2:
                    logging.info("[結果] BUY すでにポジションあり")
                elif a==1:
                    logging.info("[結果] BUY 成功")
                else:
                    logging.error("[結果] BUY 失敗")
                logging.info("[エントリー判定] BUY トレンド確定")
            elif trend == "SELL" and (macd_bearish or macd_cross_down) and sma_cross_down and adx >= 20 and rsi > 30:
                shared_state["trend"] = trend
                shared_state["trend_start_time"] = datetime.now()
                notify_slack(f"[トレンド] MACDクロスSELL（RSI={rsi_str}, ADX={adx_str}）")
                a=first_oder(trend,shared_state)
                if a==2:
                    logging.info("[結果] SELL すでにポジションあり")
                elif a==1:
                    logging.info("[結果] SELL 成功")
                else:
                    
                    logging.error("[結果] SELL 失敗")
                logging.info("[エントリー判定] SELL トレンド確定")
            else:
                shared_state["trend"] = None
                notify_slack(f"[スキップ] MACDクロス未検出のためスキップ（RSI={rsi_str}, ADX={adx_str}, MACD={macd_str}, Signal={signal_str}）")
                logging.info("[スキップ] MACDクロスなし")
        logging.info(f"[判定条件] trend={trend}, macd_cross_up={macd_cross_up}, macd_cross_down={macd_cross_down}, RSI={rsi:.2f}, ADX={adx:.2f}")
        
        if shared_state.get("cmd") == "save_adx":
            save_price_history(list(high_prices), list(low_prices), list(close_prices))
            notify_slack("[保存] 外部コマンドによりADX蓄積データを保存しました")
            shared_state["cmd"] = None  # フラグをリセット
        await asyncio.sleep(interval_sec)

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
    loop = asyncio.get_event_loop()

    # 全タスクを登録
    server_task = asyncio.create_task(start_socket_server(shared_state))
    hold_status_task = loop.create_task(monitor_hold_status(shared_state, stop_event, interval_sec=1))
    trend_task = loop.create_task(monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=shared_state))
    loss_cut_task = loop.create_task(monitor_positions_fast(shared_state, stop_event, interval_sec=1))
    quick_profit_task = loop.create_task(monitor_quick_profit(shared_state, stop_event))

    # エラー通知
    trend_task.add_done_callback(lambda t: notify_slack(f"トレンド関数が終了しました: {t.exception()}"))
    quick_profit_task.add_done_callback(lambda t: notify_slack(f"即時利確関数が終了しました: {t.exception()}"))
    # 全てのタスクを待機（終了しない限り常駐）
    await asyncio.gather(
        server_task,
        hold_status_task,
        trend_task,
        loss_cut_task,
        quick_profit_task
    )
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
            
            if positions:
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
    except SystemExit as e:
        notify_slack(f"auto_trade()が終了 {type(e).__name__}: {e}")
        
    except Exception as e:
        notify_slack(f"[致命的エラー] auto_trade() にて {type(e).__name__}: {e}")
        logging.exception("auto_tradeで例外が発生しました")
        raise  # systemdが再起動してくれるならraiseで良い
    finally:
        stop_event.set()
        trend_task.cancel()
        loss_cut_task.cancel()
        quick_profit_task.cancel()
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
            await quick_profit_task
        except asyncio.CancelledError:
            notify_slack("[INFO] monitor_quick_profit タスク終了")
if __name__ == "__main__":
    try:
        asyncio.run(auto_trade())
    except SystemExit as e:
        notify_slack(f"auto_trade()が終了 {type(e).__name__}: {e}")
    except:
        save_state(shared_state)
        save_price_buffer(price_buffer)
        