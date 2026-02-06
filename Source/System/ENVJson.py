import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
STOP_STATE_PATH = "/opt/Innovations/System/trade_stop_date.json"

REASON_LOSS = 1      # 損失
REASON_PROFIT = 2    # 利益確保

def today_jst_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")

def _atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic replace
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
from typing import Optional, Dict, Tuple
def _read_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ---- 旧「理由なし版」と同じ関数名 ----

def mark_stop_today(reason: int) -> None:
    """
    今日を停止日に設定（reason必須）
    reason: 1=損失停止, 2=利益確保停止
    """
    if reason not in (REASON_LOSS, REASON_PROFIT):
        raise ValueError("reason must be 1 (loss) or 2 (profit)")

    _atomic_write_json(STOP_STATE_PATH, {
        "date": today_jst_str(),
        "reason": int(reason),
    })

def is_stopped_today() -> Tuple[bool, Optional[int]]:
    """
    今日が停止日なら (True, reason) を返す。
    それ以外は (False, None)。
    """
    data = _read_json(STOP_STATE_PATH)
    if not data:
        return (False, None)

    if data.get("date") != today_jst_str():
        return (False, None)

    r = data.get("reason")
    if r in (REASON_LOSS, REASON_PROFIT):
        return (True, int(r))

    # reason壊れてたら停止扱いにしたいなら(True, None)でもOK
    return (True, None)

def clear_stop_date() -> None:
    """停止日ファイルを削除（手動解除したい時用）"""
    try:
        os.remove(STOP_STATE_PATH)
    except FileNotFoundError:
        pass