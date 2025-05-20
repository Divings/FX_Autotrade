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

# === ログファイルを初期化 ===
if os.path.exists("fx_debug_log.txt"):
    os.remove("fx_debug_log.txt")

# === 初期設定 ===
SYMBOL = "USD_JPY"
LOT_SIZE = 1000  # 1ロット = 10,000通貨
PROFIT_THRESHOLD = 0.03  # 利確幅（例: 0.1円）
LOSS_THRESHOLD = 0.05    # 損切り幅（例: 0.1円）
LOG_FILE = "fx_trade_log.csv"
CHECK_INTERVAL = 3  # 秒
MAINTENANCE_MARGIN_RATIO = 0.5  # 証拠金維持率アラート閾値

# === 環境変数の読み込み ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL_FX = "https://forex-api.coin.z.com/private"
FOREX_PUBLIC_API = "https://forex-api.coin.z.com/public"

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
        logging.info(f"[市場] ステータス: {status}")
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

# === 証拠金維持率取得 ===
def get_margin_status():
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
        logging.info(f"[証拠金レスポンス] {json.dumps(data, indent=2)}")
        ratio_raw = data.get("marginRatio")
        if ratio_raw is None or ratio_raw == 0:
            logging.info("[証拠金維持率] ポジションが存在しないため未算出です。※現在0またはNone")
        else:
            ratio = float(ratio_raw)
            logging.info(f"[証拠金維持率] {ratio:.2f}%")
            if ratio < MAINTENANCE_MARGIN_RATIO * 100:
                logging.warning("[⚠️アラート] 証拠金維持率が危険水準")
    except Exception as e:
        logging.error(f"[証拠金] 取得失敗: {e}")

# === 注文発行 ===
def open_order():
    path = "/v1/order"
    method = "POST"
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    body_dict = {
    "symbol": SYMBOL,
    "side": "BUY",
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
        logging.info(f"[注文] 新規建て: {res.json()}")
        return res.json()
    except Exception as e:
        logging.error(f"[注文] 新規建て失敗: {e}")
        return None

# === ポジション決済 ===
def close_order(position_id, size,side):
    logging.error(f"[TEST] side: {side}")
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
        logging.info(f"[決済] 成功: {res.json()}")
        return res.json()
    except Exception as e:
        logging.error(f"[決済] 失敗: {e}")
        return None

# === ログ記録 ===
def write_log(action, price):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "action", "price"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, price])

# === トレンド取得 ===
def detect_trend_by_ma(sample_duration_min=3, interval_sec=5, short_period=5, long_period=10):
    prices = []
    sample_count = max(long_period, (sample_duration_min * 60) // interval_sec)

    logging.info(f"[MAトレンド判定] {sample_count}回価格取得（{sample_duration_min}分間）")

    for _ in range(sample_count):
        p = get_price()
        if p:
            prices.append(p["bid"])
        time.sleep(interval_sec)

    if len(prices) < long_period:
        logging.warning("[MAトレンド判定] サンプル不足 → 判定不能")
        return None

    short_ma = sum(prices[-short_period:]) / short_period
    long_ma = sum(prices[-long_period:]) / long_period

    logging.info(f"[MAトレンド判定] 短期MA: {short_ma:.5f}, 長期MA: {long_ma:.5f}")

    diff = short_ma - long_ma
    if abs(diff) < 0.01:
        return None

    return "BUY" if diff > 0 else "SELL"

trend_none_count = 0
# === メイン処理 ===
def auto_trade():
    global trend_none_count
    while True:
        try:
            if not is_market_open():
                logging.info("[市場] 閉場中 → 待機")
                time.sleep(CHECK_INTERVAL)
                continue

            get_margin_status()
            positions = get_positions()
            prices = get_price()
            if not positions:
                # trend = detect_trend(sample_duration_min=2, interval_sec=12)

                trend = detect_trend_by_ma(sample_duration_min=2, interval_sec=5, short_period=6, long_period=13)
                if trend is None:
                    trend_none_count += 1
                    logging.warning(f"[トレンド] 判定不能（{trend_none_count}回連続）")
                    if trend_none_count >= 2:
                        logging.info("[トレンド] 判定不能が2回 → BUYでエントリー")
                        trend = "BUY"
                        trend_none_count = 0
                    else:
                        time.sleep(CHECK_INTERVAL)
                        continue
            else:
                trend_none_count = 0  # リセット
            if prices is None:
                time.sleep(CHECK_INTERVAL)
                continue

            ask = prices["ask"]
            bid = prices["bid"]

            if not positions:
                logging.info("[建玉] なし → 新規買い")
                open_order()
                write_log("BUY", ask)
            else:
                MAX_LOSS = 45
                MIN_PROFIT = 20
                close_side = None

                for pos in positions:
                    entry = float(pos["price"])
                    pid = pos["positionId"]
                    size = pos["size"]  # ← 建玉のsizeは「ロット単位の文字列」または float
                    
                    size_str = int(size)
                    
                    side = pos.get("side", "BUY").upper()  # 建玉方向
                    
                    if side == "BUY":
                        close_side = "SELL" 
                    else :
                        close_side = "BUY"
                    profit = round((bid - entry) * LOT_SIZE, 2)  # 実際の損益金額を算出

                    if profit >= MIN_PROFIT:
                        logging.info(f"[決済] 利確条件（利益が {profit} 円）→ 決済")                        
                        close_order(pid, size_str, close_side)
                        write_log("SELL", bid)
                    elif profit <= -MAX_LOSS:
                        logging.info(f"[決済] 損切り条件（損失が {profit} 円）→ 決済")
                        close_order(pid, size_str,close_side)
                        write_log("LOSS_CUT", bid)
                    else:
                        logging.info("[保有] 継続")

        except Exception as e:
            logging.error(f"[例外] 処理失敗: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    auto_trade()
