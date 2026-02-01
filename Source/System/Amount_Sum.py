import requests
import hmac
import hashlib
import time
from datetime import datetime, date,timedelta
from zoneinfo import ZoneInfo
from decimal import Decimal, InvalidOperation
import sqlite3
from pathlib import Path

def get_today_total_amount(
    api_key,
    secret_key,
    params = None,
    jst_date =None,
    return_count = False,   # Trueなら(合計, 件数)を返す
):
    end_point = "https://forex-api.coin.z.com/private"
    path = "/v1/executions"
    if params is None:
        params = {"symbol": "USD_JPY",
    "count": 100}

    JST = ZoneInfo("Asia/Tokyo")
    target_date = jst_date or datetime.now(JST).date()

    api_timestamp = f"{int(time.mktime(datetime.now().timetuple()))}000"
    method = "GET"
    text = api_timestamp + method + path
    sign = hmac.new(secret_key.encode("ascii"), text.encode("ascii"), hashlib.sha256).hexdigest()

    headers = {
        "API-KEY": api_key,
        "API-TIMESTAMP": api_timestamp,
        "API-SIGN": sign
    }

    res = requests.get(end_point + path, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()

    exec_list = data.get("data", {}).get("list") or []  # Noneでも空でもOKにする

    # 返ってこなかった（listが空）なら、0を返す
    if not exec_list:
        return (Decimal("0"), 0) if return_count else Decimal("0")

    total = Decimal("0")
    count = 0

    for item in exec_list:
        ts = item.get("timestamp")
        amt_str = item.get("amount", "0")
        if not ts:
            continue

        try:
            dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue

        dt_jst = dt_utc.astimezone(JST)
        if dt_jst.date() != target_date:
            continue

        try:
            total += Decimal(str(amt_str))
            count += 1
        except (InvalidOperation, TypeError):
            continue

    return (total, count) if return_count else total

def init_sqlite() -> sqlite3.Connection:
    DB_PATH = "daily_amount.db"
    conn = sqlite3.connect(Path(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_amount_summary (
            trade_date   TEXT NOT NULL,   -- 'YYYY-MM-DD' (JST)
            symbol       TEXT NOT NULL,   -- 例: 'USD_JPY'
            total_amount TEXT NOT NULL,   -- Decimalを文字列保存
            saved_at     TEXT NOT NULL,   -- ISO8601 (JST)
            PRIMARY KEY (trade_date, symbol)
        )
        """
    )
    conn.commit()
    return conn


def save_daily_summary(SYMBOL,total_amount: Decimal) -> None:
    JST = ZoneInfo("Asia/Tokyo")
    trade_date = datetime.now(JST).date().isoformat()
    saved_at = datetime.now(JST).isoformat()

    conn = init_sqlite()
    try:
        conn.execute(
            """
            INSERT INTO daily_amount_summary (trade_date, symbol, total_amount, saved_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(trade_date, symbol) DO UPDATE SET
                total_amount=excluded.total_amount,
                saved_at=excluded.saved_at
            """,
            (trade_date, SYMBOL, str(total_amount), saved_at)
        )
        conn.commit()
    finally:
        conn.close()

def get_yesterday_total_amount_from_sqlite(SYMBOL):
    """
    前日（JST）の total_amount だけ返す。
    無ければ None。
    ※ total_amount はDBに文字列で保存してる想定なので、戻り値も str。
    """
    JST = ZoneInfo("Asia/Tokyo")
    yesterday = (datetime.now(JST).date() - timedelta(days=1)).isoformat()

    conn = init_sqlite()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT total_amount
            FROM daily_amount_summary
            WHERE trade_date = ? AND symbol = ?
            """,
            (yesterday, SYMBOL)
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()