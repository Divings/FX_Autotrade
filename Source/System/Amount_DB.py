import mysql.connector
from decimal import Decimal
from datetime import date, timedelta
from typing import Optional
import os

from dotenv import load_dotenv

# .env読み込み
load_dotenv()

DB = dict(
            host = os.getenv('DB_HOST'),
            port = int(os.getenv('DB_PORT', 3306)),
            user = os.getenv('DB_USER'),
            password = os.getenv('DB_PASS'),
            database = os.getenv('DB_NAME')
        )

def _to_decimal(v) -> Decimal:
    if v is None:
        return Decimal("0")
    if isinstance(v, Decimal):
        return v
    return Decimal(str(v))

def upsert_daily_pnl(trade_date: date, pnl) -> None:
    """1日1行で当日決算損益を保存（同日があれば上書き）"""
    pnl = _to_decimal(pnl)

    conn = mysql.connector.connect(**DB)
    try:
        cur = conn.cursor()
        sql = """
        INSERT INTO daily_realized_pnl (trade_date, pnl)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
          pnl = VALUES(pnl)
        """
        cur.execute(sql, (trade_date, pnl))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def get_daily_pnl(trade_date: date) -> Optional[Decimal]:
    """指定日の当日決算損益をDecimalで返す。無ければNone。"""
    conn = mysql.connector.connect(**DB)
    try:
        cur = conn.cursor()
        cur.execute("SELECT pnl FROM daily_realized_pnl WHERE trade_date=%s", (trade_date,))
        row = cur.fetchone()
        return _to_decimal(row[0]) if row else None
    finally:
        cur.close()
        conn.close()

def get_today_pnl(default: Optional[Decimal] = None) -> Optional[Decimal]:
    """今日の損益。無ければdefault（未指定ならNone）。"""
    v = get_daily_pnl(date.today())
    return v if v is not None else default

def get_yesterday_pnl(default: Optional[Decimal] = None) -> Optional[Decimal]:
    """前日の損益。無ければdefault（未指定ならNone）。"""
    v = get_daily_pnl(date.today() - timedelta(days=1))
    return v if v is not None else default


if __name__ == "__main__":
    # 例: 今日の決算損益を保存
    upsert_daily_pnl(date.today(), "-1234.56")

    # 取得
    print("today:", get_today_pnl(default=Decimal("0")))
    print("yesterday:", get_yesterday_pnl(default=Decimal("0")))
