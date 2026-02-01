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
from datetime import datetime, date
from zoneinfo import ZoneInfo
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any, Union, Tuple


def get_today_total_lossgain_latest(
    api_key: str,
    secret_key: str,
    symbol: str,
    jst_date: Optional[date] = None,          # Noneなら今日(JST)
    count: int = 100,                          # latestExecutions の最大が基本100
    end_point: str = "https://forex-api.coin.z.com/private",
    return_count: bool = False,                # Trueなら(合計, 件数)を返す
    close_only: bool = False,                  # Trueなら settleType=="CLOSE" だけ合計
) -> Union[Decimal, Tuple[Decimal, int]]:
    """
    /v1/latestExecutions から当日(JST)の lossGain 合計を Decimal で返す。
    - 返ってこない（list空）なら 0
    - return_count=True なら (合計, 件数)
    - close_only=True なら決済（CLOSE）だけを集計（当日決済損益っぽくしたい場合に便利）
    """
    JST = ZoneInfo("Asia/Tokyo")
    target_date = jst_date or datetime.now(JST).date()

    path = "/v1/latestExecutions"
    method = "GET"

    # 署名用タイムスタンプ（ミリ秒）
    api_timestamp = f"{int(time.mktime(datetime.now().timetuple()))}000"

    # 署名生成
    text = api_timestamp + method + path
    sign = hmac.new(secret_key.encode("ascii"), text.encode("ascii"), hashlib.sha256).hexdigest()

    headers = {
        "API-KEY": api_key,
        "API-TIMESTAMP": api_timestamp,
        "API-SIGN": sign
    }

    # 念のため count の下限上限をクリップ（API側制限に寄せる）
    if count is None:
        count = 100
    count = int(count)
    if count <= 0:
        count = 1
    if count > 100:
        count = 100

    params: Dict[str, Any] = {
        "symbol": symbol,
        "count": count
    }

    res = requests.get(end_point + path, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    payload = res.json()

    exec_list = payload.get("data", {}).get("list") or []
    if not exec_list:
        return (Decimal("0"), 0) if return_count else Decimal("0")

    total = Decimal("0")
    matched = 0

    for item in exec_list:
        # 決済だけに限定したい場合
        if close_only and item.get("settleType") != "CLOSE":
            continue

        ts = item.get("timestamp")
        if not ts:
            continue

        # UTC(Z) -> JST
        try:
            dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue

        dt_jst = dt_utc.astimezone(JST)
        if dt_jst.date() != target_date:
            continue

        lg_str = item.get("lossGain", "0")
        try:
            total += Decimal(str(lg_str))
            matched += 1
        except (InvalidOperation, TypeError):
            continue

    return (total, matched) if return_count else total


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

