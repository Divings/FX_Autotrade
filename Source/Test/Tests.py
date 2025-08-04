from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

def fetch_usdjpy_economic_events():
    url = "https://jp.investing.com/economic-calendar/"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    #resp = requests.get(url, headers=headers)
    with open("debug.html", "w", encoding="utf-8") as f:
        f.write(resp.text)
    print("HTML保存完了。ファイルを開いて中身を確認してください。")

    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    rows = soup.select("table.economicCalendarTable tbody tr")
    events = []
    today = datetime.now().strftime("%Y-%m-%d")

    for row in rows:
        time_cell = row.find("td", class_="first left time")
        if time_cell is None:
            continue
        time_str = time_cell.get_text(strip=True)
        try:
            dt = datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            continue

        currency_tag = row.find("td", class_="left flagCur noWrap")
        if currency_tag is None:
            continue
        currency = currency_tag.get_text(strip=True)

        impact_html = row.find("td", class_="sentiment")
        impact = impact_html.get_text(strip=True).count("牛") if impact_html else 0

        event = row.find("td", class_="event").get_text(strip=True)
        print(len(rows))
        if currency not in ["USD", "JPY"]:
            continue

        events.append({
            "datetime": dt,
            "currency": currency,
            "impact": impact,
            "event": event
        })

    df = pd.DataFrame(events)
    if not df.empty:
        df["impact_level"] = df["impact"].map({3: "高", 2: "中", 1: "低", 0: "低"})
        df = df[["datetime", "currency", "impact_level", "event"]]
    return df

# テスト用実行ブロック
if __name__ == "__main__":
    df = fetch_usdjpy_economic_events()
    if df.empty:
        print("📭 指標データが取得できませんでした。")
        
    else:
        print("📅 取得された経済指標一覧:")
        print(df.to_string(index=False))
input(" >> ")