import os
import hmac
import hashlib
import json
import requests
import time
import csv
import sys
from datetime import datetime
from dotenv import load_dotenv

# === 設定 ===
SYMBOL = "USD_JPY"
LOT_SIZE = 0.1  # 1ロット = 10,000通貨。0.1は1,000通貨
PROFIT_THRESHOLD = 0.10  # 利確幅（例: 0.1円）
LOSS_THRESHOLD = 0.10    # 損切り幅（例: 0.1円）
LOG_FILE = "fx_trade_log.csv"
FULL_LOG_FILE = "fx_debug_log.txt"
CHECK_INTERVAL = 60  # 秒
MAINTENANCE_MARGIN_RATIO = 0.5  # 証拠金維持率アラート閾値（50%）
MARKET_STATUS_LOGGED = False  # 初回ログ記録制御用

# === ログの標準出力/エラーをファイルにリダイレクト ===
sys.stdout = open(FULL_LOG_FILE, "a", buffering=1)
sys.stderr = sys.stdout

# === 環境変数の読み込み ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://api.coin.z.com"
FOREX_PUBLIC_API = "https://forex-api.coin.z.com/public"

# === 署名作成 ===
def create_signature(timestamp, method, path, body=""):
    message = timestamp + method + path + body
    return hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()

# === 営業状態チェック（初回のみ記録） ===
def is_market_open():
    global MARKET_STATUS_LOGGED
    try:
        response = requests.get(f"{FOREX_PUBLIC_API}/v1/status")
        if response.status_code != 200:
            print(f"[市場] ステータスコード異常: {response.status_code}")
            return False
        status = response.json().get("status")
        if not MARKET_STATUS_LOGGED:
            print(f"[初回記録] 取引所ステータス: {status}")
            MARKET_STATUS_LOGGED = True
        return status == "OPEN"
    except Exception as e:
        print(f"[エラー] 営業状態取得に失敗しました: {e}")
        return False

# === 建玉取得 ===
def get_positions():
    path = "/private/v1/position"
    method = "GET"
    timestamp = str(int(time.time() * 1000))
    sign = create_signature(timestamp, method, path)

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign,
    }

    try:
        res = requests.get(BASE_URL + path, headers=headers)
        res.raise_for_status()
        positions = res.json().get("data", [])
        return [p for p in positions if p["symbol"] == SYMBOL]
    except Exception as e:
        print(f"[エラー] 建玉取得に失敗しました: {e}")
        return []

# === 現在価格取得（bid / ask 両方取得） ===
def get_price():
    try:
        res = requests.get(f"{FOREX_PUBLIC_API}/v1/ticker")
        res.raise_for_status()
        data = res.json().get("data", [])
        for item in data:
            if item.get("symbol") == SYMBOL:
                ask = float(item["ask"])
                bid = float(item["bid"])
                return {"ask": ask, "bid": bid}
        print(f"[エラー] 指定シンボル {SYMBOL} が見つかりませんでした。")
        return None
    except Exception as e:
        print(f"[エラー] 現在価格の取得失敗: {e}")
        return None

# === 証拠金維持率取得 ===
def get_margin_status():
    path = "/private/v1/account/margin"
    method = "GET"
    timestamp = str(int(time.time() * 1000))
    sign = create_signature(timestamp, method, path)

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    try:
        res = requests.get(BASE_URL + path, headers=headers)
        res.raise_for_status()
        data = res.json().get("data", {})
        margin_ratio = float(data.get("marginRatio", 0))
        print(f"[証拠金維持率] {margin_ratio:.2f}%")
        if margin_ratio < MAINTENANCE_MARGIN_RATIO * 100:
            print("[⚠️アラート] 証拠金維持率が危険水準です！")
    except Exception as e:
        print(f"[エラー] 証拠金維持率取得失敗: {e}")

# === 損失シミュレーション ===
def simulate_max_loss():
    price = get_price()
    if price:
        max_loss = LOSS_THRESHOLD * 1000
        print(f"[シミュレーション] 最大想定損失（1回）: 約{int(max_loss)}円")

# === 成行注文（新規建て） ===
def open_order():
    path = "/private/v1/order"
    method = "POST"
    timestamp = str(int(time.time() * 1000))
    body = json.dumps({
        "symbol": SYMBOL,
        "side": "BUY",
        "executionType": "MARKET",
        "size": str(LOT_SIZE),
        "symbolType": "FOREX"
    })
    sign = create_signature(timestamp, method, path, body)

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign,
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(BASE_URL + path, headers=headers, data=body)
        print(f"[新規建て] {res.json()}")
        return res.json()
    except Exception as e:
        print(f"[エラー] 新規注文失敗: {e}")
        return None

# === 成行決済（ポジションクローズ） ===
def close_order(position_id, size):
    path = "/private/v1/closeOrder"
    method = "POST"
    timestamp = str(int(time.time() * 1000))
    body = json.dumps({
        "positionId": position_id,
        "executionType": "MARKET",
        "size": str(size)
    })
    sign = create_signature(timestamp, method, path, body)

    headers = {
        "API-KEY": API_KEY,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign,
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(BASE_URL + path, headers=headers, data=body)
        print(f"[決済] {res.json()}")
        return res.json()
    except Exception as e:
        print(f"[エラー] 決済失敗: {e}")
        return None

# === ログ保存（売買記録） ===
def write_log(action, price):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "action", "price"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, price])

# === メインループ ===
def auto_trade():
    simulate_max_loss()
    while True:
        try:
            if not is_market_open():
                print("[市場] 市場は閉じています → 待機")
                time.sleep(CHECK_INTERVAL)
                continue

            get_margin_status()

            positions = get_positions()
            prices = get_price()
            if prices is None:
                time.sleep(CHECK_INTERVAL)
                continue

            ask_price = prices["ask"]
            bid_price = prices["bid"]

            if not positions:
                print(f"\n[情報] 建玉なし → 新規買い建て実行")
                open_order()
                write_log("BUY", ask_price)
            else:
                for pos in positions:
                    entry_price = float(pos["price"])
                    position_id = pos["positionId"]
                    size = float(pos["size"])

                    print(f"\n[情報] 建玉: ID={position_id}, 買値={entry_price:.3f}, 現在ASK={ask_price:.3f}, BID={bid_price:.3f}, 数量={size}")

                    if bid_price >= entry_price + PROFIT_THRESHOLD:
                        print("[情報] 利確条件達成 → 決済実行")
                        close_order(position_id, size)
                        write_log("SELL", bid_price)
                    elif bid_price <= entry_price - LOSS_THRESHOLD:
                        print("[情報] 損切り条件達成 → 決済実行")
                        close_order(position_id, size)
                        write_log("LOSS_CUT", bid_price)
                    else:
                        print("[情報] 条件未達 → 継続保有")

        except Exception as e:
            print(f"[例外エラー] {e}")

        time.sleep(CHECK_INTERVAL)

# === 実行 ===
if __name__ == "__main__":
    auto_trade()
