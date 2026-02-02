# -*- coding: utf-8 -*-
import os
import requests
import time
import configparser
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from conf_load import load_settings_from_db

# .env を読み込む（TELEGRAM_TOKEN / TELEGRAM_CHAT_ID を使うため）
load_dotenv()

import base64
from pathlib import Path
from Crypto.Cipher import AES

KEY_FILE = Path("aes_key.bin")

# AESキーをファイルから読み込む
def load_aes_key():
    if not KEY_FILE.exists():
        raise RuntimeError("AESキーが見つかりません (aes_key.bin)")
    return KEY_FILE.read_bytes()

_AES_KEY = load_aes_key()

# AES-GCM で復号
def aes_decrypt(token: str) -> str:
    raw = base64.b64decode(token)
    nonce = raw[:16]
    tag = raw[16:32]
    ciphertext = raw[32:]
    cipher = AES.new(_AES_KEY, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode()

# 設定読み込み
def load_config():
    """
    config.ini の settings セクションから debug / Setdefault を読み込み
    Setdefault は 'slack' または 'telegram'
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    debug = False
    default = "slack"
    try:
        debug = config.getboolean('settings', 'debug')
    except Exception:
        pass
    try:
        default = config.get('settings', 'Setdefault', fallback="slack")
    except Exception:
        pass
    return debug, default

debug, default_service = load_config()

# Slack Webhook URL（暗号化済み）
config1 = load_settings_from_db()

# Slack Webhook URL の復号
_slack_enc = config1.get("SLACK_WEBHOOK_URL")

# 復号処理
if _slack_enc:
    try:
        SLACK_WEBHOOK_URL = aes_decrypt(_slack_enc)
    except Exception as e:
        raise RuntimeError("SLACK_WEBHOOK_URL の復号に失敗しました") from e
else:
    SLACK_WEBHOOK_URL = None

# Telegram トークン（必須）
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
# chat_id は未設定なら自動取得し、取得後に .env へ追記してキャッシュする
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 連投抑止（クールダウン）
_last_notify_times = {}
_NOTIFY_COOLDOWN_SECONDS = 60

# 直近送信メッセージのハッシュ（メモリ）
msg_history = None

# --- 直近ハッシュの永続化（プロセス再起動後も重複判定可能） ---
_HASH_FILE = "notification_hash.txt"
_LOG_FILE  = "notification_log.txt"

from typing import Optional

# ハッシュファイルから直近ハッシュを読み込み
def _read_last_hash_from_file(path: str = _HASH_FILE) -> Optional[str]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            return line or None
    except Exception:
        return None

# ハッシュファイルへ直近ハッシュを書き込み
def _write_last_hash_to_file(h: str, path: str = _HASH_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(h + "\n")
    except Exception as e:
        # 保存失敗は致命ではないのでログだけ
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as lf:
                lf.write(f"[ハッシュ保存失敗] {e}（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）\n")
        except Exception:
            pass
# ------------------------------------------------------------

# .env へ key=value を追記（既存キーは上書き、なければ追記）
def _append_env_if_needed(key: str, value: str, env_path: str = ".env"):
    """ .env に key=value を追記（既存キーは上書き、なければ追記） """
    try:
        if not os.path.exists(env_path):
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"{key}={value}\n")
            return

        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                updated = True
                break
        if not updated:
            lines.append(f"{key}={value}")

        with open(env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines and not lines[-1].endswith("\n") else ""))
    except Exception as e:
        print(f"[ENV書き込み警告] {key} の保存に失敗: {e}")

values=0 #エラー時の通知抑止用変数

# メッセージに応じた色分け
def _message_color_for_slack(message: str) -> str:
    """ 元ロジックの色分け """
    if "[即時損切]" in message or "[決済] 損切り" in message or "[⚠️アラート]" in message:
        return "#ff4d4d"  # 赤
    elif "[決済]" in message or "[即時利確]" in message or "[RSI" in message:
        return "#36a64f"  # 緑
    elif "[保有]" in message:
        return "#439FE0"  # 青
    elif "[建玉]" in message or "[スプレッド]" in message:
        return "#daa520"  # オレンジ
    elif "[エラー]" in message or "[注意]" in message:
        return "#8b0000"  # 暗赤
    elif "[INFO]" in message:
        return "#888888"  # グレー
    else:
        return "#dddddd"  # 薄グレー

# Telegram chat_id を取得・保存
def _get_telegram_chat_id() -> str:
    """ TELEGRAM_CHAT_ID が未設定なら getUpdates で自動取得して .env に保存 """
    global TELEGRAM_CHAT_ID
    if TELEGRAM_CHAT_ID:
        return TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN is not set. .env に TELEGRAM_TOKEN を設定してください。")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    res = requests.get(url)
    try:
        data = res.json()
    except Exception:
        raise RuntimeError(f"chat_id の取得に失敗しました（JSON 変換不可）: {res.text[:200]}")

    results = data.get("result", [])
    if not results:
        raise RuntimeError(
            "getUpdates に結果がありません。Bot に /start を送ってから再実行してください。"
            "（Webhook 有効時は getUpdates が空になるため、必要なら deleteWebhook を）"
        )
    last = results[-1]
    chat = None
    if "message" in last and "chat" in last["message"]:
        chat = last["message"]["chat"]
    elif "channel_post" in last and "chat" in last["channel_post"]:
        chat = last["channel_post"]["chat"]
    elif "edited_message" in last and "chat" in last["edited_message"]:
        chat = last["edited_message"]["chat"]
    if not chat or "id" not in chat:
        raise RuntimeError("getUpdates の応答に chat.id が見つかりません。")

    TELEGRAM_CHAT_ID = str(chat["id"])
    _append_env_if_needed("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
    return TELEGRAM_CHAT_ID

# Slack 通知実装
def _notify_slack_impl(message: str):
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL is not set.")
    color = _message_color_for_slack(message)
    payload = {"attachments": [{"color": color, "text": message}]}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
    if response.status_code != 200:
        print(f"[Slack通知失敗] {response.status_code} - {response.text}")

# Telegram 通知実装
def _notify_telegram_impl(message: str):
    chat_id = _get_telegram_chat_id()
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload, timeout=10)
    if response.status_code != 200:
        print(f"[Telegram通知失敗] {response.status_code} - {response.text}")

# ログファイル追記
def _append_log(reason: str, message: str):
    """ ログファイルに理由＋本文を追記 """
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{reason}] {ts}\n{message}\n\n")
    except Exception:
        pass

# 既存互換のエントリポイント
def notify_slack(message: str):
    """
    既存互換のエントリポイント。
    config.ini の settings:Setdefault に従って Slack または Telegram へ送信する。
    直前と同じメッセージは「通知せず」、メッセージ本文を notification_log.txt へ追記。
    クールダウン抑止時も本文をログへ残す。
    """
    global values
    global msg_history
    time.sleep(1.2)  # 連続送信対策のインターバル

    now = time.time()
    msg_hash = hashlib.sha256(message.encode()).hexdigest()

    # 直前と同一か（メモリ or ファイルの記録で判定）
    file_hash = _read_last_hash_from_file()
    is_same_as_memory = (msg_hash == msg_history)
    is_same_as_file = (file_hash is not None and msg_hash == file_hash)
    if is_same_as_memory or is_same_as_file:
        _append_log("重複抑止(直前と同一)", message)
        return

    # クールダウン抑止（ハッシュではなくメッセージ文字列で管理）
    last_sent = _last_notify_times.get(message)
    if last_sent and (now - last_sent < _NOTIFY_COOLDOWN_SECONDS):
        _append_log("クールダウン抑止", message)
        return

    _last_notify_times[message] = now

    # Debug モードなら目印を付与（重複判定には影響させないため、この位置で付与）
    send_text = "[Debug モード] " + message if debug else message

    try:
        if str(default_service).lower() == "telegram":
            _notify_telegram_impl(send_text)
        else:
            _notify_slack_impl(send_text)

        # 送信成功後に「元メッセージ」のハッシュを更新（Debug 付与前の本文で計算したもの）
        msg_history = msg_hash
        _write_last_hash_to_file(msg_hash)

    except Exception as e:
        if values==0: #エラー時の通知抑止
            print(f"[通知例外] {e}")
        else:
            values=1
    except ValueError:
        if values==0: #エラー時の通知抑止
            print(f"[通知例外] {e}")
        else:
            values=1