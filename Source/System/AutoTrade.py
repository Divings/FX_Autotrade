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
from decimal import Decimal, ROUND_HALF_UP
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
from Assets import assets

# ミッドナイトモード(Trueで有効化)
night = True
MAX_Stop = 180
SYS_VER = "3.5.6"

import numpy as np

def build_last_n_candles_from_prices(prices: list[float], n: int = 20) -> list[dict]:
    """
    prices から直近 n 本のローソク足を構築
    1本あたり20ティックで構成
    """
    ticks_per_candle = 20
    max_candles = len(prices) // ticks_per_candle
    candles_to_build = min(n, max_candles)

    if candles_to_build == 0:
        logging.warning("データが不足しています。ローソク足を生成できません。")
        return []

    logging.info(f"price_bufferの長さ: {len(prices)} / 作れるローソク足: {candles_to_build}")

    candles = []

    for i in range(candles_to_build):
        end = len(prices) - i * ticks_per_candle
        start = max(0, end - ticks_per_candle)
        slice = prices[start:end]
        if not slice:
            continue
        candle = {
            "open": slice[0],
            "close": slice[-1],
            "high": max(slice),
            "low": min(slice),
        }
        candles.insert(0, candle)  # 時系列順にするため先頭に挿入

    return candles

def calculate_range(price_buffer, period=10):
    candles = build_last_n_candles_from_prices(list(price_buffer), n=period)

    if not candles:
        # ローソク足が1本も作れなければ None
        return None

    actual_period = min(period, len(candles))
    highs = [candle['high'] for candle in candles[-actual_period:]]
    lows  = [candle['low'] for candle in candles[-actual_period:]]

    return max(highs) - min(lows)

def calculate_dmi(highs, lows, closes, period=14):
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)

    plus_dm = np.zeros_like(highs)
    minus_dm = np.zeros_like(lows)

    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]

        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))

    plus_di = 100 * (np.convolve(plus_dm[1:], np.ones(period), 'valid') / period) / (np.convolve(tr, np.ones(period), 'valid') / period)
    minus_di = 100 * (np.convolve(minus_dm[1:], np.ones(period), 'valid') / period) / (np.convolve(tr, np.ones(period), 'valid') / period)

    return plus_di, minus_di

import os
import shutil
import requests
from EncryptSecureDEC import decrypt_file

import statistics

import platform
if platform.python_version() != "3.9.21":
    notify_slack("エラー:動作保証バージョンを満たしていません")
    sys.exit(1)

def is_volatile(prices, candles, period=5):
    import statistics
    from collections import deque

    if not isinstance(prices, (list, tuple, deque)) or len(prices) < period + 10:
        return False
    if not isinstance(candles, list) or len(candles) < period + 10:
        return False

    if isinstance(prices, deque):
        prices = list(prices)

    try:
        recent_prices = prices[-period:]
        stdev_value = statistics.stdev(recent_prices)

        # 過去の中央値と比べてボラが高いか判断
        historical_stdevs = [statistics.stdev(prices[i - period:i]) for i in range(period, period + 10)]
        median_stdev = statistics.median(historical_stdevs)
        dynamic_threshold_stdev = median_stdev * 1.2  # ←過去より20%高ければボラ高

    except statistics.StatisticsError:
        return False

    # 動的しきい値でチェック
    if stdev_value > dynamic_threshold_stdev:
        return True

    # ヒゲ比率チェック（直近のローソク足）
    last = candles[-1]
    body = abs(last["open"] - last["close"])
    high = last["high"]
    low = last["low"]
    wick_upper = high - max(last["open"], last["close"])
    wick_lower = min(last["open"], last["close"]) - low
    wick_ratio = (wick_upper + wick_lower) / (body + 1e-5)  # 0除算回避

    avg_candle_size = statistics.mean([c["high"] - c["low"] for c in candles[-10:]])
    dynamic_wick_ratio_threshold = 2.0 if avg_candle_size < 0.5 else 1.0

    if wick_ratio > dynamic_wick_ratio_threshold:
        return True

    # 高低差によるチェック
    highlow_diff = high - low
    avg_highlow = statistics.mean([c["high"] - c["low"] for c in candles[-10:]])
    if highlow_diff > avg_highlow * 1.5:
        return True

    return False  # 安定


def download_two_files(base_url, download_dir):
    filenames = ["API.txt.vdec", "SECRET.txt.vdec"]
    
    for filename in filenames:
        url = f"{base_url}/{filename}"
        download_path = os.path.join(download_dir, filename)
        
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download file {filename}: {response.status_code}")
        
        with open(download_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        # print(f"Downloaded {filename} to {download_path}")
        
import os
import shutil
import requests
import lzma
import hashlib
import json
import datetime
import getpass
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2

BLOCKCHAIN_HEADER = b'BLOCKCHAIN_DATA_START\n'

def write_README(temp_dir,path,message):
    if path!=None:
        save_dir = temp_dir + path +"README.txt"
    else:
        save_dir = temp_dir + "/README.txt"
    # ファイルに保存
    with open(save_dir, "w", encoding="utf-8") as f:
        f.write(message)

txt_message="このディレクトリは各種ログが記録されます。\nシステム再起動の原因となるため、手動取引を行う場合あらかじめシステムを停止してください。\nシステムの再起動により発生したすべての損害を開発者は補償しません\n"

def write_info(id,temp_dir):
    save_dir = temp_dir + "/log/" + str(id) + "_order_info.json"
    endPoint  = 'https://forex-api.coin.z.com/private'
    path      = '/v1/orders'
    method    = 'GET'
    timestamp = str(int(time.time() * 1000))  # ミリ秒タイムスタンプ

    text = timestamp + method + path
    sign = hmac.new(
        API_SECRET.encode('ascii'),
        text.encode('ascii'),
        hashlib.sha256
    ).hexdigest()

    parameters = { "rootOrderId": id }

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.get(endPoint + path, headers=headers, params=parameters)

    try:
        response_data = res.json()
        formatted_json = json.dumps(response_data, indent=2)

        # ファイルに保存
        with open(save_dir, "a", encoding="utf-8") as f:
            f.write(formatted_json)
            f.write("\n")

        # print("[保存完了] order_info.json に書き込みました")

    except json.decoder.JSONDecodeError:
        print("[エラー] サーバーからの応答がJSON形式ではありません")
        # print(res.text)

# ダウンロード関数はそのまま
def download_two_files(base_url, download_dir):
    filenames = ["API.txt.vdec", "SECRET.txt.vdec"]
    for filename in filenames:
        url = f"{base_url}{filename}"
        download_path = os.path.join(download_dir, filename)
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download file {filename}: {response.status_code}")
        with open(download_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        # print(f"Downloaded {filename} to {download_path}")

# API読み込み関数
def load_api(temp_dir):
    
    # パスワードを.envから読み込み
    password = os.getenv("API_PASSWORD")
    password2 = os.getenv("SECRET_PASSWORD")
    if not password or not password2:
        raise Exception("環境変数 API_PASSWORD または SECRET_PASSWORD が設定されていません")

    download_two_files(URL_Auth, temp_dir)

    # 復号処理
    file_path1 = os.path.join(temp_dir, "API.txt.vdec")
    decrypted_path1 = decrypt_file(file_path1, password)

    file_path2 = os.path.join(temp_dir, "SECRET.txt.vdec")
    decrypted_path2 = decrypt_file(file_path2, password2)

    # 復号済ファイル読み取り
    with open(decrypted_path1, 'r', encoding='utf-8') as f:
        api_data = f.read()

    with open(decrypted_path2, 'r', encoding='utf-8') as f:
        secret_data = f.read()

    # 復号後のファイルは削除
    os.remove(file_path1)
    os.remove(file_path2)
    os.remove(decrypted_path1)
    os.remove(decrypted_path2)

    return api_data, secret_data

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
    "oders_error":False,
    "last_skip_hash":None,
    "cooldown_untils":None,
    "firsts":False,
    "max_profit":None,  # 保有中に記録された最大利益
    "trail_offset":20
}

import configparser
def load_ini():
    try:
        # ConfigParser オブジェクトを作成
        config = configparser.ConfigParser()
        # config.ini を読み込む
        config.read('config.ini')
        reset = config.getboolean('settings', 'reset')
    except:
        reset = False
    return reset

reset = load_ini()
args=sys.argv
file_path = sys.argv[0]  # スクリプトファイルのパス
folder_path = os.path.dirname(os.path.abspath(file_path))
os.chdir(folder_path)

import tempfile

temp_dir = tempfile.mkdtemp()
os.makedirs(temp_dir + "/" + "log", exist_ok=True)
key_box = tempfile.mkdtemp()
session = requests.Session() # セッションを生成

txt_message="このシステムはInnovation Craft Inc.の所有物です。\n正規の手段、手順以外で得たコードを使用した場合、法的措置の対象となる場合があります。\n\n"
write_README(temp_dir,"/log/",txt_message)
write_README(temp_dir,None,txt_message)
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

def is_trend_initial(candles, min_body_size=0.003, min_breakout_ratio=0.003):
    """
    ローソク足リスト（最低2本）から初動を判定（緩め）
    """
    if shared_state.get("cooldown_untils", 0) > time.time():
        # まだクールタイム中
        notify_slack("[スキップ] クールタイム中なので初動判定をスキップ")
        return False, ""

    if len(candles) < 2:
        return False, ""

    # 末尾の2本を使う
    prev = candles[-2]
    last = candles[-1]

    body_last = abs(last["close"] - last["open"])
    body_prev = abs(prev["close"] - prev["open"])
    range_prev = prev["high"] - prev["low"]
    range_last = last["high"] - last["low"]

    # 最低実体サイズチェック
    if body_last < min_body_size:
        return False, ""

    # 買いの初動
    if (
        last["close"] > prev["high"] and
        (last["close"] - last["open"]) > body_prev and
        last["close"] > last["open"] and
        (last["close"] - prev["high"]) >= min_breakout_ratio
    ):
        return True, "BUY"

    # 売りの初動
    if (
        last["close"] < prev["low"] and
        (last["open"] - last["close"]) > body_prev and
        last["close"] < last["open"] and
        (prev["low"] - last["close"]) >= min_breakout_ratio
    ):
        return True, "SELL"

    return False, ""

# ===ログ設定 ===
LOG_FILE1 = f"{temp_dir}/fx_debug_log.txt"
try:
    _log_last_reset = datetime.now()
except:
    _log_last_reset = datetime.datetime.now()
os.makedirs("last_temp", exist_ok=True)
now = datetime.datetime.now()

# フォーマット
formatted = now.strftime("%Y/%m/%d %H:%M")
with open(f"last_temp/last_temp.txt", "w", encoding="utf-8") as f:
    f.write(f"最終記録 {formatted} \n")
    f.write(temp_dir)
    f.write("\n")

import platform
os_name = platform.system()

if os_name=="Windows":
    print(temp_dir)

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
if shared_state.get("cmd") == "save_adx":
    shared_state["cmd"] == None

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

def reverse_side(side: str) -> str:
    return "SELL" if side.upper() == "BUY" else "BUY"

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
        etime=shared_state.get("entry_time")
        for pos in positions:
            pid = pos["positionId"]
            elapsed = time.time() - etime
            entry = float(pos["price"])
            size = int(pos["size"])
            side = pos.get("side", "BUY").upper()
            MAX_HOLD = 300
            EXTENDABLE_LOSS = -10  # 許容する微損（円）
            profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

            if elapsed > MAX_HOLD:
                if shared_state.get("firsts")==True:
                    logging.info("延長 保有時間超過だが初動検知のためスキップ")
                    return
                if profit > EXTENDABLE_LOSS and shared_state.get("trend") == side:
                    logging.info("[延長] 保有時間超過だがトレンド継続中のため保持")
                    return  # 決済せず延長
                else:
                    notify_slack(f"注意! 保有時間が長すぎます\n 強制決済を発動します {profit}")
                    rside = reverse_side(side)
                    close_order(pid, size, rside)
                    record_result(profit, shared_state)
                
            # 通知条件：利益または損失が±10円以上、かつ通知内容が前回と違うとき
            if abs(profit) > 10:
                prev = last_notified.get(pid)
                if prev is None or abs(prev - profit) >= 5:  # 5円以上変化時のみ再通知
                    notify_slack(f"[保有] 建玉{pid} 継続中: {profit}円")
                    last_notified[pid] = profit
        await asyncio.sleep(interval_sec)

def should_skip_entry(candles, direction: str, recent_resistance=None, recent_support=None, atr=None, min_atr=0.05):
    """
    BUY or SELL エントリー直前にスキップすべきかどうかを判定する関数
    改善版：トレンド方向、高値/安値水準、ボラティリティも考慮

    Args:
        candles (list[dict]): 過去のローソク足（最低2本必要）
        direction (str): "BUY" または "SELL"
        recent_resistance (float): 直近の高値ゾーン
        recent_support (float): 直近の安値ゾーン
        atr (float): 現在のATR値
        min_atr (float): 最低限のボラティリティしきい値

    Returns:
        (bool, str): (スキップすべきか, 理由メッセージ)
    """
    last = candles[-1]
    prev = candles[-2]

    open1, close1 = prev["open"], prev["close"]
    high1, low1 = prev["high"], prev["low"]
    open2, close2 = last["open"], last["close"]
    high2, low2 = last["high"], last["low"]

    def body(o, c): return abs(o - c)

    # ボラティリティが低すぎる
    if atr is not None and atr < min_atr:
        return True, f"ボラティリティ不足（ATR={atr:.3f} < {min_atr}） → 見送り"

    if direction == "BUY":
        # 直前足が陰線
        if close1 < open1:
            return True, "直前足が陰線 → BUY見送り"

        # 上ヒゲが長い（失速）
        upper_wick = high1 - max(open1, close1)
        if upper_wick > body(open1, close1):
            return True, "上ヒゲ優勢 → BUY見送り"
        
        range1 = candles[-1]["high"] - candles[-1]["low"]
        range2 = candles[-2]["high"] - candles[-2]["low"]
        avg_range = (range1 + range2) / 2
        tolerance = avg_range * 0.3
        
        # 高値ゾーンに近い（過去n本の最高値に近い）
        if recent_resistance is not None and abs(high1 - recent_resistance) < tolerance:
            return True, "高値ゾーン接近 → BUY見送り"
       
    elif direction == "SELL":
        # 直前足が陽線
        if close1 > open1:
            return True, "直前足が陽線 → SELL見送り"
     
        # 安値ゾーンに到達
        if recent_support is not None and low1 <= recent_support:
            return True, "安値ゾーンで長いヒゲ → SELL見送り"

    return False, ""

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
            
            ask = Decimal(str(ask))
            entry = Decimal(str(entry))
            bid = Decimal(str(bid))
            # LOT_SIZE = Decimal(str(LOT_SIZE))

            # 利益計算
            raw_profit = (ask - entry if side == "BUY" else entry - bid) * LOT_SIZE
            profit = raw_profit.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

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
                if shared_state.get("firsts")==True:
                    shared_state["cooldown_untils"] = time.time() + MAX_Stop
                    shared_state["firsts"] = False
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
    # deque, list, tuple のいずれかか確認
    if not isinstance(prices, (list, tuple, deque)) or len(prices) < 2:
        return False

    # dequeの場合はリストに変換
    if isinstance(prices, deque):
        prices = list(prices)

    try:
        return statistics.stdev(prices[-5:]) > threshold
    except statistics.StatisticsError:
        return False

import copy
Buffer = copy.copy(MAX_LOSS)
_PREV_MAX_LOSS = None

def adjust_max_loss(prices,
                    base_loss=50,
                    vol_thresholds=(0.03, 0.05),
                    adjustments=(5, 10),
                    period=5):
    """
    ボラティリティに応じて MAX_LOSS を調整してグローバルに設定
    前回と値が変わったときのみログ・Slack通知
    """
    global MAX_LOSS, _PREV_MAX_LOSS

    if len(prices) < period:
        MAX_LOSS = base_loss
        if MAX_LOSS != _PREV_MAX_LOSS:
            msg = f"[INFO] データ不足のため MAX_LOSS = {MAX_LOSS}円"
            logging.info(msg)
            notify_slack(msg)
            _PREV_MAX_LOSS = MAX_LOSS
        return

    vol = statistics.stdev(list(prices)[-period:])

    if vol > vol_thresholds[1]:
        new_max_loss = base_loss + adjustments[1]
    elif vol > vol_thresholds[0]:
        new_max_loss = base_loss + adjustments[0]
    else:
        new_max_loss = Buffer

    MAX_LOSS = new_max_loss

    # 値が変わったときだけ通知
    if MAX_LOSS != _PREV_MAX_LOSS:
        msg = f"[INFO] ボラ={vol:.4f}, MAX_LOSS を更新: {MAX_LOSS}円"
        logging.info(msg)
        notify_slack(msg)
        _PREV_MAX_LOSS = MAX_LOSS

def handle_exit(signum, frame):
    print("SIGTERM 受信 → 状態保存")
    save_state(shared_state)
    save_price_buffer(price_buffer)  
    sys.exit(0)

# === 環境変数の読み込み ===
conf=load_settings_from_db()
URL_Auth = conf["URL"]
api_data, secret_data=load_api(temp_dir)

API_KEY = api_data.strip()
API_SECRET = secret_data.strip()
BASE_URL_FX = "https://forex-api.coin.z.com/private"
FOREX_PUBLIC_API = "https://forex-api.coin.z.com/public"

out = assets(API_KEY,API_SECRET)
try:
    available_amounts = out['data']['availableAmount']
    available_amount = int(float(out['data']['availableAmount']))
    notify_slack(f"現在の取引余力は{available_amount}円です。")
except:
    pass

if os.path.isfile("pricesData.txt") == False and now.hour>=1:
    with open("pricesData.txt", "w", encoding="utf-8") as f:
        f.write(available_amounts)
else:
    with open("pricesData.txt", "r", encoding="utf-8") as f:
        saved_available_amounts = f.read().strip()
        try:
            saved_available_amount = float(saved_available_amounts)
        except ValueError:
            logging.error("基準初期残高読み込み時にエラー")
            saved_available_amount = out['data']['availableAmount']

from AddData import insert_data
def last_balance():
    SECRET_KEYs = os.getenv("SECRET_PASSWORD").encode()
    global available_amounts
    if os.path.isfile("pricesData.txt") == True:
        with open("pricesData.txt", "r", encoding="utf-8") as f:
            saved_available_amounts = f.read().strip()
            try:
                saved_available_amount = float(saved_available_amounts)
            except ValueError:
                logging.error("基準初期残高読み込み時にエラー")
                saved_available_amount = out['data']['availableAmount']
    else:
        saved_available_amount = out['data']['availableAmount']
    
    out = assets(API_KEY,API_SECRET)
    last = round(float(out['data']['availableAmount']) - float(saved_available_amount), 2)
    sign1 = hmac.new(SECRET_KEYs, str(last).encode(), hashlib.sha256).hexdigest()
    notify_slack(f"[当日決算損益] 当日決算損益は{last}円です。")
    available_amounts = out['data']['availableAmount'] # 定数を更新
    result = insert_data(
        table="Same-day-profit",
        columns=["Profit", "sign"],
        values=(str(last), sign1)
    )
    if result:
        logging.info("データ挿入成功")
    else:
        logging.error("データ挿入失敗")

    with open("pricesData.txt", "w", encoding="utf-8") as f:
        f.write(available_amounts)
    return

import os
import requests
import subprocess
import sys

# パラメータ設定
PUBLIC_KEY_URL = URL_Auth + "key/publickey.asc"
PUBLIC_KEY_FILE = "/opt/gpg/publickey.asc"
UPDATE_FILE = "AutoTrade.py"
SIGNATURE_FILE = "AutoTrade.py.sig"

def download_public_key(url, save_path):
    """公開鍵をダウンロードして保存"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        # print("公開鍵をダウンロードしました")
    except Exception as e:
        notify_slack(f"公開鍵ダウンロード失敗: {str(e)}")
        sys.exit(1)

def import_public_key(gpg_home, key_path):
    """公開鍵をGPGにインポート"""
    try:
        subprocess.run(['gpg', '--homedir', gpg_home, '--import', key_path], check=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print("公開鍵をインポートしました")
    except subprocess.CalledProcessError:
        notify_slack("公開鍵インポート失敗")
        sys.exit(1)

def verify_signature(gpg_home, signature_file, update_file):
    """署名検証"""
    result = subprocess.run(
        ['gpg', '--homedir', gpg_home, '--verify', signature_file, update_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        notify_slack(f"署名検証失敗！起動中止:{update_file}")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    notify_slack("[INFO] 署名検証成功")

def notify_asset():
    out=assets(API_KEY,API_SECRET)
    available_amount = int(float(out['data']['availableAmount']))
    balance = int(float(out['data']['balance']))

    notify_slack(f"現在の取引余力は{available_amount}円です。\n 現在の現金残高は{balance}円です。")
    return 0

# EncryptSecureDECの署名検証
public_key_path = os.path.join(key_box, "publickey.asc")
download_public_key(PUBLIC_KEY_URL, public_key_path)
import_public_key(key_box, public_key_path)
# verify_signature(key_box, "EncryptSecureDEC.py.sig", "EncryptSecureDEC.py")

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

rootOrderIds = None
# === ポジション決済 ===
def close_order(position_id, size, side):
    global rootOrderIds
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
            if rootOrderIds != None:
                logging.info(f"ID {rootOrderIds}を決済")
                write_info(rootOrderIds,temp_dir)
            rootOrderIds = None
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

def first_order(trend,shared_state=None):
    # now = datetime.datetime.now()
    global rootOrderIds
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
                data = open_order(trend)
                if data and "data" in data and "rootOrderId" in data["data"]:
                    rootOrderIds = data["data"][0].get("rootOrderId")
                    if rootOrderIds != None:
                        logging.info(f"ID {rootOrderIds}を注文")
                        write_info(rootOrderIds,temp_dir)
                else:
                    rootOrderIds = None
                    # notify_slack("[エラー] 注文のrootOrderIdが取得できませんでした")
                shared_state["entry_time"] = time.time()
                write_log(trend, ask)
                return 1
            except Exception as e:
                notify_slack(f"[注文失敗] {e}")
                return 0
    else:
        return 2

def failSafe():
    """もし終了前に建玉があった時用"""
    positions = get_positions()
    prices=get_price()
    if positions:
        for pos in positions:
            entry = float(pos["price"])
            pid = pos["positionId"]
            size_str = int(pos["size"])
            side = pos.get("side", "BUY").upper()
            close_side = "SELL" if side == "BUY" else "BUY"
            close_order(pid,size_str,close_side)
            bid = prices["bid"]
            write_log("LOSS_CUT", bid)
    else:
        print("強制決済建玉なし")
        return 0
    
def build_last_2_candles_from_prices(prices: list[float]) -> list[dict]:
    """
    price_buffer（1秒〜数秒おきの価格履歴）から直近2本のローソク足を構築
    1分あたり20本程度の粒度と仮定
    """
    if len(prices) < 40:
        return []

    candles=[]
    logging.info(f"price_bufferの長さ: {len(price_buffer)}")
    for i in range(2):
        start = -40 + i*20
        end = None if i == 1 else start + 20
        slice = prices[start:end]
        if not slice:
            continue
        candle = {
            "open": slice[0],
            "close": slice[-1],
            "high": max(slice),
            "low": min(slice),
        }
        candles.append(candle)

    return candles

async def process_entry(trend, shared_state, price_buffer,rsi_str,adx_str,candles):
    shared_state["trend"] = trend
    shared_state["trend_start_time"] = datetime.datetime.now()
    notify_slack(f"[トレンド] MACDクロス{trend}（RSI={rsi_str}, ADX={adx_str}）")

    #candles = build_last_2_candles_from_prices(list(price_buffer))
    #if len(price_buffer) < 40:
    #    logging.info("価格履歴がまだ不足しているので待機")
    #    return
    if not candles or len(candles) < 2:
        logging.info(candles)
        logging.error("ローソク足データが不足しているためスキップ")
        notify_slack("ローソク足データが不足しているためスキップ")
        return
    skip, reason = should_skip_entry(candles, trend)

    if skip:
        shared_state["trend"] = None
        logging.info(f"[エントリースキップ] {reason}")
        notify_slack(f"[スキップ] {reason}")
        await asyncio.sleep(3)
    else:
        a = first_order(trend, shared_state)
        if a == 2:
            logging.info(f"[結果] {trend} すでにポジションあり")
        elif a == 1:
            logging.info(f"[結果] {trend}  取引 成功")
            shared_state["last_trend"] = trend
        else:
            logging.error(f"[結果] {trend} 失敗")
        logging.info(f"[エントリー判定] {trend} トレンド確定")

def dynamic_filter(adx, rsi, bid, ask):
    now = datetime.datetime.now()
    hour = now.hour

    # スプレッドの計算
    spread = ask - bid

    # 時間帯によってしきい値を変更
    if 9 <= hour < 15:
        adx_threshold = 35
        spread_threshold = 0.25
    elif 15 <= hour < 22:
        adx_threshold = 25
        spread_threshold = 0.3
    else:
        adx_threshold = 20
        spread_threshold = 0.35

    # 各条件のチェック
    if spread > spread_threshold:
        logging.info(f"[スキップ] スプレッド過大: {spread:.3f} > {spread_threshold}")
        return False

    if adx < adx_threshold:
        logging.info(f"[スキップ] ADX不足: {adx:.1f} < {adx_threshold}")
        return False

    if rsi <= 20 or rsi >= 80:
        logging.info(f"[スキップ] RSI過熱/過冷: {rsi:.1f}")
        return False

    return True

candle_buffer = []
# === トレンド判定を拡張（RSI+ADX込み） ===
async def monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=None):
    import statistics
    from collections import deque
    from datetime import datetime
    from datetime import date
    import time
    import logging
    global candle_buffer
    global price_buffer
    # price_buffer = deque(maxlen=240)
    global MAX_SPREAD
    high_prices, low_prices, close_prices = load_price_history()
    xstop = 0
    trend = shared_state.get("trend",None)
    
    VOL_THRESHOLD_SHORT = 0.006
    VOL_THRESHOLD_LONG = 0.008
    import hashlib
    last_notified = {}  # 建玉ごとの通知済みprofit記録
    max_profits = {}    # 建玉ごとの最大利益記録
    TRAILING_STOP = 15
    
    last_rsi_state = None
    last_adx_state = None
    sstop = 0
    vstop = 0
    nstop = 0
    timestop = 0
    m = 0
    s = 0
    last = 0
    n_nonce = 0
    m_note = 0
    nn_nonce = 0
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
        status_market = is_market_open()
        if status_market != "OPEN" or weekday_number == 6 or weekday_number == 5:
            if sstop == 0:
                notify_slack(f"[市場] 市場が{status_market}中")
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
        if now.hour == 0 and now.minute == 0:
            if last == 0:
                last_balance()
                last = 1
        elif last == 1:
            last = 0
            
        if now.hour < 4:
            high_prices.clear()
            low_prices.clear()
            close_prices.clear()
            price_buffer.clear()
            m = 0
            shared_state["price_reset_done"] = True 
        if now.hour == 6:
            if s == 0:
                notify_asset()
                s = 1
            if s == 1 and now.hour !=6:
                s = 0
        if night != True:
            midnight = False
            if now.hour >= TIME_STOP or now.hour < 5:
                midnight = False
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
        else:
            now = datetime.now()
            if now.hour <= 4:
                # 誤差吸収のため値を初期化
                high_prices.clear()
                low_prices.clear()
                close_prices.clear()
                price_buffer.clear()
                shared_state["price_reset_done"] = True 
            if now.hour >= 21:
                midnight = True
                m = 0
            elif now.hour >= 4:
                midnight = False
                m = 1
            if m == 0 and midnight==False:
                notify_slack(f"[INFO]ミッドナイトモードが有効です。\n 夜間取引を行います、市場の状況により大きな損失が発生する場合があります。")
                m = 1

        from datetime import datetime, timezone

        now =  datetime.now(timezone.utc)

        # 日本時間（UTC+9）に換算
        hour_jst = (now.hour + 9) % 24

        if 9 <= hour_jst <= 22:
            MAX_SPREAD = 0.03  # 日中は今のまま
        else:
            MAX_SPREAD = 0.007  # 夜間は厳しめ

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

        short_ma = sum(list(price_buffer)[-short_period:]) / short_period
        long_ma = sum(list(price_buffer)[-long_period:]) / long_period
        
        sma_cross_up = short_ma > long_ma and shared_state.get("last_short_ma", 0) <= shared_state.get("last_long_ma", 0)
        sma_cross_down = short_ma < long_ma and shared_state.get("last_short_ma", 0) >= shared_state.get("last_long_ma", 0)
        logging.info(f"price_bufferの長さ: {len(price_buffer)}")
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
        #macd_bearish = macd[-1] < signal[-1]  # デッドクロスまたは継続中

        if adx is not None:
            if adx < 20:
                TRAILING_STOP = 10
            elif adx < 40:
                TRAILING_STOP = 15
            else:
                TRAILING_STOP = 25
        else:
            TRAILING_STOP = 15
            
        positions = get_positions()
        if positions:
            prices = get_price()
            ask = prices["ask"]
            bid = prices["bid"]

            for pos in positions:
                pid = pos["positionId"]
                side = pos.get("side", "BUY").upper()
                entry = float(pos["price"])
                size = int(pos["size"])
                elapsed = time.time() - shared_state.get("entry_time", time.time())
        
                prices = get_price()
                ask = prices["ask"]
                bid = prices["bid"]
        
                # 現在利益の計算
                profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

                # 最大利益の更新
                if pid not in max_profits:
                    max_profits[pid] = profit
                elif profit > max_profits[pid]:
                    max_profits[pid] = profit
                if profit > 0:
                    logging.info(f"[トレール更新] 建玉{pid} 現在の最大利益更新: {profit}円")

                # トレーリングストップ判定
                if profit <= max_profits[pid] - TRAILING_STOP:
                    notify_slack(f"[トレーリングストップ] 建玉{pid} 最大利益{max_profits[pid]}円 → 利益確保して決済")
                    close_order(pid, size, reverse_side(side))
                    record_result(profit, shared_state)
                    # 削除する前に確認
                    if pid in max_profits:
                        del max_profits[pid]

        macd_str = f"{macd[-1]:.5f}" if macd[-1] is not None else "None"
        signal_str = f"{signal[-1]:.5f}" if signal[-1] is not None else "None"

        rsi_limit = (trend == "BUY" and rsi < 70) or (trend == "SELL" and rsi > 30)
        logging.info(f"[MACD] クロス判定: UP={macd_cross_up}, DOWN={macd_cross_down}")
        logging.info(f"[判定詳細] trend候補={trend}, diff={diff:.5f}, stdev={statistics.stdev(list(price_buffer)[-5:]):.5f}")
        
        candles = build_last_2_candles_from_prices(list(price_buffer))
        #candles = candle_buffer[-1:] + candles
        # if len(candle_buffer) > 50:
        #    candle_buffer=candle_buffer[-50]
        logging.info(f"[INFO] キャンドルデータ2本分 {candles}")
        range_value = calculate_range(price_buffer, period=10)
        
        if range_value != None:
            if adx >= 20 and range_value >= 0.05:
                if nn_nonce == 0:
                    notify_slack(f"[横ばい判定] 価格が動き始めました")
                    logging.info("[スキップ] 価格が動き始め")
                    nn_nonce = 1
                    shared_state["cooldown_untils"] = time.time() + MAX_Stop
            else:
                trend = None
                nn_nonce = 0
                shared_state["trend"] = None
                if n_nonce == 0:
                    notify_slack(f"[横ばい判定] 価格変動幅が小さいためスキップ")
                    logging.info("[スキップ] 価格横ばい")
                    n_nonce = 1
                await asyncio.sleep(interval_sec)
                continue
        
        if len(close_prices) >= 5:
            price_range = max(close_prices) - min(close_prices)
            if price_range < 0.03:
                trend = None
                shared_state["trend"] = None
                notify_slack(f"[横ばい判定] 価格変動幅が小さい（{price_range:.4f}）ためスキップ")
                logging.info("[スキップ] 価格横ばい")
                await asyncio.sleep(interval_sec)
                continue
        
        if midnight == True and adx < 50:
            if m_note == 0 and now.hour >= 21:
                notify_slack(f"[INFO] ミッドナイトモード中だが、ADXが低いのでスキップ ADX={adx_str}")
                m_note = 1
            continue
        else:
            m_note = 0

        today_str = datetime.now().strftime("%Y-%m-%d")
        if adx >= 95:
            # 無効化（非常事態）
            shared_state["trend"] = None
            notify_slack(f"[警告] ADXが100に近いためスキップ（ADX={adx:.2f}）")
            logging.warning("[スキップ] ADX異常値 → 判定中止")
            continue
        elif adx >= 70 and abs(diff) > 0.015 and trend is not None and (now.hour > 5 and now.hour < 9):
            last_forced_entry_date = shared_state.get("forced_entry_date")

            if last_forced_entry_date == today_str:
                logging.info("[強制エントリー制限] 本日すでに実行済みのためスキップ")
                
            else:
                now = datetime.now()
                if now.hour <= 21 or now.hour <= 5:
                    timestop = 1
                    # クロス不要で許可
                    shared_state["trend"] = trend
                    try:
                        notify_slack(f"[強トレンド] MACDクロス無視してエントリー（ADX={adx:.2f}, diff={diff:.4f}）")
                    except:
                        pass
                    notify_slack(f"[トレンド] MACDクロス{trend}（RSI={rsi_str}, ADX={adx_str}）")
                    a=first_order(trend,shared_state)
                    if a==2:
                        logging.info(f"[結果] {trend} すでにポジションあり")
                    elif a==1:
                        logging.info(f"[結果] {trend} 成功")
                        shared_state["oders_error"]=False
                        shared_state["forced_entry_date"] = today_str
                    else:
                        logging.error(f"[結果] {trend} 失敗")
                    if a==1:
                        logging.info("[エントリー] ADX強すぎるためクロス無視")
                        shared_state["forced_entry_date"] = today_str
                        shared_state["last_trend"] = trend
                else:
                    if timestop == 0:
                        if now.hour <=5 :
                            notify_slack(f"[情報] MACDクロス無視してエントリーだが、5時前なのでスキップ")
                            logging.info("[情報] MACDクロス無視してエントリーだが、5時前なのでスキップ")
                        elif now.hour >=22:
                            notify_slack(f"[情報] MACDクロス無視してエントリーだが、9時以降なのでスキップ")
                            logging.info("[情報] MACDクロス無視してエントリーだが、9時以降なのでスキップ")
                        timestop = 1
        n_nonce = 0
        if rsi < 20:
            notify_slack(f"[RSI下限] RSI={rsi_str} → 反発警戒でスキップ")
            logging.info("[スキップ] RSI下限で警戒")
            await asyncio.sleep(interval_sec)
            continue
        short_stdev = statistics.stdev(list(price_buffer)[-5:])
        long_stdev = statistics.stdev(list(price_buffer)[-20:])
        
        
        is_initial, direction = is_trend_initial(candles)
        if is_initial:
            # 簡易フィルター
            positions = get_positions()
            if not positions:
        
                # BUY or SELL によって RSI のしきい値を設定
                rsi_ok = True
                if direction == "BUY" and rsi >= 70:
                    rsi_ok = False
                if direction == "SELL" and rsi <= 30:
                    rsi_ok = False
                
                if is_high_volatility(close_prices) == False:
                    msg = f"[スキップ] {trend} ボラティリティ低のためエントリースキップ"
                    logging.info(msg)
                    notify_slack(msg)
                    continue

                if spread < MAX_SPREAD and adx >= 20 and rsi_ok:
                    logging.info(f"初動検出、方向: {direction} → エントリー")
                    notify_slack(f"初動検出、方向: {direction} → エントリー")
                    first_order(direction, shared_state)
                    direction = None
                    is_initial = None
                    shared_state["trend"] = None
                    shared_state["cooldown_untils"] = time.time() + MAX_Stop
                    shared_state["firsts"] = True
                else:
                    logging.info(f"初動だが条件未達 → 見送り (spread={spread}, adx={adx}, rsi={rsi})")
            else:
                logging.info(f"建玉あり → エントリーせず")
        else:
            logging.info("初動ではない")

        if short_stdev > VOL_THRESHOLD_SHORT and long_stdev > VOL_THRESHOLD_LONG:
            
            trend = "BUY" if diff > 0 else "SELL"
            if adx < 20:
                notify_slack(f"[スキップ] ADXが低いためトレンド弱くスキップ（ADX={adx:.2f}）")
                shared_state["trend"] = None
                await asyncio.sleep(interval_sec)
                continue
            
            TREND_HOLD_MINUTES = 15  # 任意の継続時間

            now = datetime.now()
            adjust_max_loss(close_prices)
            trend_active = False
            if is_volatile(close_prices, candles):
                notify_slack("[フィルター] 乱高下中につき判定スキップ")
                continue  # トレンド判定処理を一時スキップ
            # ここにDMI判定を追加する
            plus_di, minus_di = calculate_dmi(high_prices, low_prices, close_prices)
            current_plus_di = plus_di[-1]
            current_minus_di = minus_di[-1]

            # DMI方向一致判定
            dmi_trend_match = False
            if trend == "BUY" and current_plus_di > current_minus_di:
                dmi_trend_match = True
            elif trend == "SELL" and current_minus_di > current_plus_di:
                dmi_trend_match = True
            
            logging.info(f"[INFO] DMI TREND {dmi_trend_match}")

            if "trend_start_time" in shared_state:
                elapsed = (now - shared_state["trend_start_time"]).total_seconds() / 60.0
                if elapsed < TREND_HOLD_MINUTES:
                    trend_active = True
                    if trend != None:
                        logging.info(f"[継続中] {shared_state['trend']}トレンド継続中 ({elapsed:.1f}分経過)")
            stc = dynamic_filter(adx, rsi, bid, ask)
            if stc and trend == "BUY" and (macd_bullish or macd_cross_up) and sma_cross_up and rsi_limit and dmi_trend_match:
                if is_high_volatility(close_prices):
                    msg = f"[スキップ] {trend} ボラティリティ高のためエントリースキップ"
                    logging.info(msg)
                    notify_slack(msg)
                    continue  # エントリーしない
                await process_entry(trend, shared_state, price_buffer,rsi_str,adx_str,candles)
            elif stc and trend == "SELL" and (macd_cross_down) and sma_cross_down and rsi > 35 and rsi_limit and dmi_trend_match and statistics.stdev(list(price_buffer)[-5:]) >= 0.007 and statistics.stdev(list(price_buffer)[-20:]) >= 0.010:
                if is_high_volatility(close_prices):
                    msg = f"[スキップ] {trend} ボラティリティ高のためエントリースキップ"
                    logging.info(msg)
                    notify_slack(msg)
                    continue  # エントリーしない
                await process_entry(trend, shared_state, price_buffer, rsi_str,adx_str,candles)
            elif positions and trend == "SELL" and (macd_bullish or macd_cross_up) or trend == "BUY" and (macd_cross_down):
                notify_slack(f"[トレンド] トレンド反転 即時損切り")
                positions = get_positions()
                prices = get_price()
                if prices is None:
                    await asyncio.sleep(interval_sec)
                    continue
                if positions:
                    ask = prices["ask"]
                    bid = prices["bid"]

                    for pos in positions:
                        entry = float(pos["price"])
                        pid = pos["positionId"]
                        size_str = int(pos["size"])
                        side = pos.get("side", "BUY").upper()
                        close_side = "SELL" if side == "BUY" else "BUY"
                    close_order(pid, size_str, close_side)
                    write_log(close_side, bid)
                    
            elif positions and trend == "BUY" and macd_cross_down:
                notify_slack(f"[トレンド] トレンド反転 即時損切り")
                positions = get_positions()
                prices = get_price()
                if prices is None:
                    await asyncio.sleep(interval_sec)
                    continue
                if positions:
                    ask = prices["ask"]
                    bid = prices["bid"]

                    for pos in positions:
                        entry = float(pos["price"])
                        pid = pos["positionId"]
                        size_str = int(pos["size"])
                        side = pos.get("side", "BUY").upper()
                        close_side = "SELL" if side == "BUY" else "BUY"
                    close_order(pid, size_str, close_side)
                    write_log(close_side, bid)
            else:
                    shared_state["trend"] = None

                    ng_reasons = []
                    if trend == "BUY":
                        macd_ok = (macd_bullish or macd_cross_up)
                        sma_ok = sma_cross_up
                        rsi_ok = (rsi < 70)
                        dmi_ok = dmi_trend_match
                        stdev_ok = True  # BUY側はstdev条件なし

                    elif trend == "SELL":
                        macd_ok = macd_cross_down
                        sma_ok = sma_cross_down
                        rsi_ok = (rsi > 35)
                        dmi_ok = dmi_trend_match
                        stdev_ok = (short_stdev >= 0.007 and long_stdev >= 0.010)

                    if not macd_ok:
                        ng_reasons.append("MACD")
                    if not sma_ok:
                        ng_reasons.append("SMA")
                    if not rsi_ok:
                        ng_reasons.append("RSI")
                    if not dmi_ok:
                        ng_reasons.append("DMI")
                    if trend == "SELL" and not stdev_ok:
                        ng_reasons.append("ボラ")
                    if ng_reasons:
                        notify_message = f"[スキップ] {trend}側 条件未達: {', '.join(ng_reasons)}"
    
                        # SHA256計算
                        hash_digest = hashlib.sha256(notify_message.encode()).hexdigest()
    
                        if hash_digest != shared_state.get("last_skip_hash"):
                            notify_slack(notify_message)
                            shared_state["last_skip_hash"] = hash_digest
                        else:
                            logging.info("[スキップ] 同一理由でスキップ → 通知抑制")
    
                        shared_state["trend"] = None
                    else:
                        shared_state["last_skip_hash"] = None  # エントリー成功時はリセット
                        notify_slack(f"[スキップ] {trend}側 条件未達: {', '.join(ng_reasons)}")
        else:
            if  (trend is not None) and trend != "BUY" and trend != "SELL":
                notify_slack(f"[スキップ] trend未定義（不明な分岐）{trend}")
        logging.info(f"[判定条件] trend={trend}, macd_cross_up={macd_cross_up}, macd_cross_down={macd_cross_down}, RSI={rsi:.2f}, ADX={adx:.2f}")
        
        if shared_state.get("cmd") == "save_adx":
            save_price_history(list(high_prices), list(low_prices), list(close_prices))
            notify_slack("[保存] 外部コマンドによりADX蓄積データを保存しました")
            shared_state["cmd"] = None  # フラグをリセット
        await asyncio.sleep(interval_sec)

# == 即時利確監視用タスク ==
async def monitor_quick_profit(shared_state, stop_event, interval_sec=1):
    global MAX_Stop
    PROFIT_BUFFER = 5  # 利確ラインに対する安全マージン
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

            ask = Decimal(str(ask))
            entry = Decimal(str(entry))
            bid = Decimal(str(bid))

            # 利益計算
            raw_profit = (ask - entry if side == "BUY" else entry - bid) * LOT_SIZE
            profit = raw_profit.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            # profit = round((ask - entry if side == "BUY" else entry - bid) * LOT_SIZE, 2)

            entry_time = shared_state.get("entry_time")
            if entry_time is None:
                continue
            elapsed = time.time() - entry_time

            # 利確ライン（スリッページ考慮）
            short_term_target = 10 + PROFIT_BUFFER
            long_term_target = 30 + PROFIT_BUFFER
            bid = prices["bid"]
            ask = prices["ask"]

            spread = ask - bid
            
            if profit <= (-MAX_LOSS + SLIPPAGE_BUFFER):
                if spread > MAX_SPREAD:
                    notify_slack(f"[即時利確保留] 強制決済実行の条件に達したが、スプレッドが拡大中なのでスキップ\n 損切/利確タイミングに注意")
                    continue
            if (elapsed <= 60 and profit >= short_term_target) or (elapsed > 60 and profit >= long_term_target):
                notify_slack(f"[即時利確] 利益が {profit} 円（{elapsed:.1f}秒保持）→ 決済実行")

                start = time.time()
                close_order(pid, size_str, close_side)
                end = time.time()

                record_result(profit, shared_state)
                write_log("QUICK_PROFIT", bid)
                if shared_state.get("firsts")==True:
                    shared_state["cooldown_untils"] = time.time() + MAX_Stop
                    shared_state["firsts"] = False
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
    server_task.add_done_callback(lambda t: notify_slack(f"情報保存用サーバが終了しました: {t.exception()}"))
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
            status_market = is_market_open()
            if  status_market != "OPEN":
                if vstop==0:
                    notify_slack(f"[市場] 市場が{status_market}中")
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
        try:
            failSafe()
        except:
            pass
        shutil.rmtree(temp_dir)
        shutil.rmtree(key_box)
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
    
    # import_public_key(key_box, public_key_path)
    verify_signature(key_box, SIGNATURE_FILE, UPDATE_FILE)
    try:
        asyncio.run(auto_trade())
    except SystemExit as e:
        notify_slack(f"auto_trade()が終了 {type(e).__name__}: {e}")
        save_state(shared_state)
        save_price_buffer(price_buffer)
    except:
        save_state(shared_state)
        save_price_buffer(price_buffer)
        