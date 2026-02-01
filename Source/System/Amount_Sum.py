import requests
import hmac
import hashlib
import time
from datetime import datetime, date
from zoneinfo import ZoneInfo
from decimal import Decimal, InvalidOperation

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