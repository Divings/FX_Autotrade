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

import requests
import hmac
import hashlib
import time
from datetime import datetime, date
from zoneinfo import ZoneInfo
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional, Tuple

JST = ZoneInfo("Asia/Tokyo")


def _to_decimal(x) -> Decimal:
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError):
        return Decimal("0")


def _ts_to_jst_date(ts: str) -> Optional[date]:
    if not ts:
        return None
    try:
        dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt_utc.astimezone(JST).date()


def sum_lossgain_today_from_api(
    api_key: str,
    secret_key: str,
    symbol: str,
    target_date: Optional[date] = None,    # Noneなら「今日(JST)」
    close_only: bool = True,              # 当日決済損益なら True 推奨
    include_fee: bool = False,            # 手数料も足すなら True
    count: int = 100,
    end_point: str = "https://forex-api.coin.z.com/private",
) -> Tuple[Decimal, int]:
    """
    /v1/latestExecutions をAPIで取得して、
    timestamp をJSTに変換した target_date（デフォルト:今日JST）の行だけ集計する。
    戻り値: (合計Decimal, 対象件数)
    """
    if target_date is None:
        target_date = datetime.now(JST).date()

    # count は最大100想定
    count = int(count)
    if count < 1:
        count = 1
    if count > 100:
        count = 100

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

    params: Dict[str, Any] = {"symbol": symbol, "count": count}

    res = requests.get(end_point + path, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    payload = res.json()

    items = payload.get("data", {}).get("list") or []

    total = Decimal("0")
    matched = 0

    for item in items:
        # 日付（JST）でフィルタ
        d = _ts_to_jst_date(item.get("timestamp"))
        if d != target_date:
            continue

        # 決済だけ
        if close_only and item.get("settleType") != "CLOSE":
            continue

        total += _to_decimal(item.get("lossGain", "0"))
        if include_fee:
            total += _to_decimal(item.get("fee", "0"))

        matched += 1

    return total, matched

def sum_yesterday_realized_pnl_at_midnight(
    api_key: str,
    secret_key: str,
    symbol: str,
    target_date: Optional[date] = None,
    close_only: bool = True
) -> Tuple[Decimal, int]:
    pnl, cnt = sum_lossgain_today_from_api(
        api_key=api_key,
        secret_key=secret_key,
        symbol=symbol,
        close_only=close_only,
        include_fee=False,
        target_date=target_date
    )
    return pnl, cnt

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

def get_yesterday_total_amount_from_sqlite(SYMBOL,mode=False):
    """
    前日（JST）の total_amount だけ返す。
    無ければ None。
    ※ total_amount はDBに文字列で保存してる想定なので、戻り値も str。
    """
    JST = ZoneInfo("Asia/Tokyo")
    yesterday = (datetime.now(JST).date() - timedelta(days=1)).isoformat()
    if mode ==True:
        today = datetime.now(JST).date()
        # weekday(): 月曜=0, 火=1, 水=2, 木=3, 金=4, 土=5, 日=6
        days_since_thursday = (today.weekday() - 3) % 7
        last_thursday = today - timedelta(days=days_since_thursday)
        yesterday = last_thursday.isoformat()

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

