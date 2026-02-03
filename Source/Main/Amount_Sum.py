import requests
import hmac
import hashlib
import time
from datetime import datetime, date,timedelta
from zoneinfo import ZoneInfo
from decimal import Decimal, InvalidOperation
import sqlite3
from pathlib import Path

import requests
import hmac
import hashlib
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Tuple
import json
from typing import Optional

def sum_yesterday_realized_pnl_at_midnight(
    api_key: str,
    secret_key: str,
    symbol: str,
    count: int = 100,
    end_point: str = "https://forex-api.coin.z.com/private",
    test_json_path: Optional[str] = None,  # ★追加：テスト時にJSONファイルを使う
) -> Tuple[Decimal, int]:
    """
    00:00に呼ぶ前提で、/v1/latestExecutions の生データから
    settleType == "CLOSE" の lossGain を合計して返す。

    日付フィルターなし（00:00〜6:00は取引しない前提のため）
    戻り値: (合計Decimal, 対象件数int)
    """
    # ----------------------------
    # ★テスト：ファイルから payload を読む
    # ----------------------------
    if test_json_path:
        with open(test_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        path = "/v1/latestExecutions"
        method = "GET"

        api_timestamp = f"{int(time.mktime(datetime.now().timetuple()))}000"
        text = api_timestamp + method + path
        sign = hmac.new(secret_key.encode("ascii"), text.encode("ascii"), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": api_key,
            "API-TIMESTAMP": api_timestamp,
            "API-SIGN": sign
        }

        count = int(count)
        if count < 1:
            count = 1
        if count > 100:
            count = 100

        params: Dict[str, Any] = {"symbol": symbol, "count": count}

        res = requests.get(end_point + path, headers=headers, params=params, timeout=30)
        res.raise_for_status()
        payload = res.json()

    items = payload.get("data", {}).get("list") or []

    total = Decimal("0")
    matched = 0

    for item in items:
        # 決済だけ
        if item.get("settleType") != "CLOSE":
            continue

        try:
            total += Decimal(str(item.get("lossGain", "0")))
            matched += 1
        except (InvalidOperation, TypeError):
            continue

    return total, matched

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

