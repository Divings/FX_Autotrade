import requests
import time
import datetime
import tempfile
import os

def record_price_data(symbol="USD_JPY", interval_sec=1):
    api_url = "https://forex-api.coin.z.com/public/v1/ticker"

    # 一時ディレクトリ作成
    
    

    filename = "price_log.csv"

    # 初期作成（ヘッダー行）
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write("timestamp,ask,bid,spread,status\n")

    while True:
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        now = datetime.datetime.now()

        #if now.hour <= 22 and now.hour >= 5:
        #    time.sleep(interval_sec)
        #     continue

        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            found = False
            for item in data:
                if item.get("symbol") == symbol:
                    ask = float(item["ask"])
                    bid = float(item["bid"])
                    spread = ask - bid
                    line = f"{timestamp},{ask},{bid},{spread},OK\n"
                    print(line.strip())
                    found = True
                    break
            if not found:
                line = f"{timestamp},,,,'{symbol} not found'\n"
        except Exception as e:
            line = f"{timestamp},,,,'Error: {str(e)}'\n"
            print(f"エラー発生: {e}")

        with open(filename, mode="a", encoding="utf-8") as f:
            f.write(line)

        time.sleep(interval_sec)

# 実行
record_price_data()
