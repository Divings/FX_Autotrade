import csv
from datetime import datetime, timedelta, time

#CSV_PATH = "news.csv"

# 指標前後のブロック幅（分）
BLOCK_BEFORE_MIN = 15
BLOCK_AFTER_MIN = 15

def init(path):
    from pathlib import Path
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

CSV_PATH=init("datas")/"news.csv"

def load_news_blocks(target_date: datetime.date):
    blocks = []

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"] != target_date.isoformat():
                continue

            hh, mm = map(int, row["time"].split(":"))
            event_dt = datetime.combine(target_date, time(hh, mm))

            start = event_dt - timedelta(minutes=BLOCK_BEFORE_MIN)
            end = event_dt + timedelta(minutes=BLOCK_AFTER_MIN)

            blocks.append((start, end, row["currency"], row["importance"]))

    return blocks


def is_blocked(now: datetime, blocks):
    for start, end, currency, importance in blocks:
        if start <= now <= end:
            return True, start, end, currency, importance
    return False, None, None, None, None

