import csv
from datetime import datetime, timedelta, time
import os
#CSV_PATH = "news.csv"

# 指標前後のブロック幅（分）
BLOCK_BEFORE_MIN = 15
BLOCK_AFTER_MIN = 15

def init(path):
    from pathlib import Path
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_block_minutes(importance: int) -> int:
    """
    指標重要度に応じたブロック時間（分）を返す
    """
    if importance >= 3:
        return 30
    return 15

# ニュース指標のCSVファイルパス
CSV_PATH=init("datas")/"news.csv"

# ニュース指標のブロック時間を読み込む
def load_news_blocks(target_date: datetime.date):
    blocks = []

    if not os.path.exists(CSV_PATH):
        return blocks

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"] != target_date.isoformat():
                continue

            hh, mm = map(int, row["time"].split(":"))
            event_dt = datetime.combine(target_date, time(hh, mm))

            importance = int(row["importance"])
            block_min = get_block_minutes(importance)

            start = event_dt - timedelta(minutes=block_min)
            end = event_dt + timedelta(minutes=block_min)

            blocks.append((
                start,
                end,
                row["currency"],
                importance
            ))

    return blocks

def is_blocked(now: datetime, blocks):
    for start, end, currency, importance in blocks:
        if start <= now <= end:
            return True, start, end, currency, importance
    return False, None, None, None, None

