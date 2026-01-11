import sqlite3
from datetime import datetime

# SQLiteデータベースファイル名
DB_NAME = "balance.db"

# データベース初期化
def init_db():
    """データベースとテーブルを作成"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weekly_balance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        week_id TEXT NOT NULL,      -- 例: 2026-W02
        weekday INTEGER NOT NULL,   -- 0=月曜, 6=日曜
        timestamp DATETIME NOT NULL,
        balance REAL NOT NULL,
        UNIQUE (week_id, weekday)
    )
    """)

    conn.commit()
    conn.close()

# 残高登録関数
def register_weekly_balance(balance: float):
    """
    月曜日・日曜日のみ残高を登録
    同じ週・同じ曜日は1回のみ
    """
    now = datetime.now()
    weekday = now.weekday()  # 月=0, 日=6

    # 月曜・日曜以外は何もしない
    if weekday not in (0, 6):
        return

    week_id = now.strftime("%Y-W%W")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO weekly_balance
        (week_id, weekday, timestamp, balance)
        VALUES (?, ?, ?, ?)
    """, (week_id, weekday, now, balance))

    conn.commit()
    conn.close()

