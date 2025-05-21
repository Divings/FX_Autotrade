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
from setlog import log_trade_to_db

notify_slack("自動売買システム起動")

# === 初期設定 ===
SYMBOL = "USD_JPY"
LOT_SIZE = 1000  # 1ロット = 10,000通貨
MAX_SPREAD = 0.03 # 許容スプレッド
MAX_LOSS = 20    # ロスカット/損切(円)
MIN_PROFIT = 40  # 利確(円)
LOG_FILE = "fx_trade_log.csv"
CHECK_INTERVAL = 3 # 秒
MAINTENANCE_MARGIN_RATIO = 0.5  # 証拠金維持率アラート閾値

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
    "last_spread":None  # ← これも追加
}

# === 環境変数の読み込み ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL_FX = "https://forex-api.coin.z.com/private"
FOREX_PUBLIC_API = "https://forex-api.coin.z.com/public"

# === トレンド判定関数 ===
from collections import deque

# monitor_trend() の外で共有してもOK（必要に応じて）
price_buffer = deque(maxlen=240)  # 12分間保存
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

async def monitor_trend(stop_event, short_period=3, long_period=5, interval_sec=3, shared_state=None):
    while not stop_event.is_set():
        # print(shared_state)
        p = get_price()
        if p:
            price_buffer.append(p["bid"])  # 過去データに追加

        if len(price_buffer) < long_period:
                if not shared_state.get("trend_init_notice"):
                    notify_slack("[MAトレンド判定] データ蓄積中 → 判定保留中")
                    shared_state["trend_init_notice"] = True
                    await asyncio.sleep(interval_sec)
                    continue
        else:
            short_ma = sum(list(price_buffer)[-short_period:]) / short_period
            long_ma  = sum(list(price_buffer)[-long_period:]) / long_period
            prev_short = shared_state.get("last_short_ma")
            prev_long  = shared_state.get("last_long_ma")

            short_ma_diff = abs(short_ma - prev_short) if prev_short is not None else 999
            long_ma_diff  = abs(long_ma - prev_long) if prev_long is not None else 999
            
            
            if short_ma_diff > 0.03 or long_ma_diff > 0.03:
            
                notify_slack(f"[MAトレンド判定] 短期MA: {short_ma:.5f}, 長期MA: {long_ma:.5f}")
                shared_state["last_short_ma"] = short_ma
                shared_state["last_long_ma"] = long_ma

            diff = short_ma - long_ma
            
            if abs(diff) < 0.03:
                shared_state["trend"] = None
                if not shared_state.get("last_skip_notice", False):
                    notify_slack("[MAトレンド判定] 差が小さく方向不明 → スキップ")
                    shared_state["last_skip_notice"] = True
                    continue
            else:
                
                shared_state["last_skip_notice"] = False
                trend = "BUY" if diff > 0 else "SELL"
                shared_state["trend"] = trend
                if shared_state.get("last_trend") != trend:
                    notify_slack(f"[MAトレンド判定] → トレンド方向は {trend}")
                    shared_state["last_trend"] = trend
                    shared_state["trend"] = trend
                    #shared_state["last_skip_notice"] = False
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

# === トレンド取得 ===
def detect_trend_by_ma(
    sample_duration_min=3, interval_sec=5, short_period=5, long_period=10,
    last_trend_holder={"trend": None}
):
    prices = []
    sample_count = max(long_period, (sample_duration_min * 60) // interval_sec)

    # 通知省略: 毎回通知せず、ログファイルやprintなどに変更してもOK
    # print(f"[MA判定] {sample_count}回価格取得中（{sample_duration_min}分間）")

    for _ in range(sample_count):
        p = get_price()
        if p:
            prices.append(p["bid"])
        time.sleep(interval_sec)

    if len(prices) < long_period:
        notify_slack("[MA判定] サンプル不足 → 判定不能")
        return None

    short_ma = sum(prices[-short_period:]) / short_period
    long_ma  = sum(prices[-long_period:]) / long_period

    # 通知は変化時だけに限定
    diff = short_ma - long_ma
    if abs(diff) < 0.01:
        return None

    current_trend = "BUY" if diff > 0 else "SELL"

    if last_trend_holder["trend"] != current_trend:
        notify_slack(f"[MAトレンド判定] → トレンド変化: {last_trend_holder['trend']} → {current_trend}")
        last_trend_holder["trend"] = current_trend

    return current_trend


from threading import Event
trend_none_count = 0
# === メイン処理 ===
stop_event = Event()

# メイン取引処理
async def auto_trade():
    global trend_none_count
    c = 0

    trend_task = asyncio.create_task(monitor_trend(stop_event, short_period=6, long_period=13, interval_sec=3, shared_state=shared_state))
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
            if last_spread is not None and abs(spread - last_spread) < 0.001:
                continue  # ほぼ変化なし → 通知しない

            shared_state["last_spread"] = spread

            if spread > MAX_SPREAD:
                notify_slack(f"[スプレッド] {spread:.3f}円 → スプレッドが広すぎるため見送り")
                
            else:
                shared_state["last_spread"] = None  # 通常状態に戻したい場合
                continue
          
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
                    elif profit <= -MAX_LOSS:
                        notify_slack(f"[決済] 損切り条件（損失が {profit} 円）→ 決済")
                        close_order(pid, size_str, close_side)
                        write_log("LOSS_CUT", bid)
                    else:
                        if abs(profit) > 10:
                            notify_slack(f"[保有] 継続 {profit}円")

            await asyncio.sleep(CHECK_INTERVAL)

    finally:
        stop_event.set()
        trend_task.cancel()
        try:
            await trend_task
        except asyncio.CancelledError:
            notify_slack("[INFO] monitor_trend タスク終了")

if __name__ == "__main__":
    asyncio.run(auto_trade())