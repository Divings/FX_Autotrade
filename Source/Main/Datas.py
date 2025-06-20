import requests
import time
import datetime
import os

def read_temp_dir(filepath="last_temp/last_temp.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 2行目がtemp_dirパス
    temp_dir = lines[1].strip()
    print(f"取得した保存先: {temp_dir}")
    return temp_dir

def is_market_open():
    try:
        response = requests.get(f"https://forex-api.coin.z.com/public/v1/status")
        response.raise_for_status()
        status = response.json().get("data", {}).get("status")
        
        return status
    except Exception as e:
        return False

def record_price_data(symbol="USD_JPY", interval_sec=1):
    api_url = "https://forex-api.coin.z.com/public/v1/ticker"
    
    # temp_dirをファイルから取得
    temp_dir = read_temp_dir()

    # 保存先ファイルパス
    filename = os.path.join(temp_dir, "price_log.csv")

    # ヘッダー行作成（既存なら上書き）
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write("timestamp,ask,bid,spread,status\n")

    while True:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        now = datetime.datetime.now()
        status = is_market_open()

        weekday_num = now.weekday()
        if now.hour >=5 or (weekday_num == 5 or weekday_num == 6 and status != "OPN"):
            break

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

# 実行例
record_price_data()
