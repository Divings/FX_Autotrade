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

# 設定をDBから読み込む（Slack Webhook は DB 側にある前提）
config1 = load_settings_from_db()
SLACK_WEBHOOK_URL = config1.get("SLACK_WEBHOOK_URL")

# Telegram トークン（必須）
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
# chat_id は未設定なら自動取得し、取得後に .env へ追記してキャッシュする
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 通知クールダウン（同一メッセージの連投抑止）
_last_notify_times = {}
_NOTIFY_COOLDOWN_SECONDS = 60
msg_history = None


def _append_env_if_needed(key: str, value: str, env_path: str = ".env"):
    """
    .env に key=value を追記（既に同じキー行があれば追記しない）
    """
    try:
        if not os.path.exists(env_path):
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"{key}={value}\n")
            return

        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        # 既存キーの行があれば上書き（なければ追記）
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


def _message_color_for_slack(message: str) -> str:
    """
    元コードの色分けロジックを踏襲
    """
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
        return "#dddddd"  # 既定（薄グレー）


def _get_telegram_chat_id() -> str:
    """
    TELEGRAM_CHAT_ID が未設定なら getUpdates で自動取得して .env に保存
    取得には『Bot に対して /start を送っていること』が前提
    """
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

    try:
        # 最新のメッセージのチャットIDを使用
        results = data.get("result", [])
        if not results:
            raise RuntimeError(
                "getUpdates に結果がありません。Bot に /start を送ってから再実行してください。"
                "（Webhook 有効時は getUpdates が空になるため、必要なら deleteWebhook を）"
            )
        last = results[-1]
        # メッセージかチャネル投稿か等で構造が異なる場合があるため順に見る
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
    except Exception as e:
        raise RuntimeError(f"chat_id の取得に失敗しました: {e}")


def _notify_slack_impl(message: str):
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL is not set.")
    color = _message_color_for_slack(message)
    payload = {"attachments": [{"color": color, "text": message}]}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
    if response.status_code != 200:
        print(f"[Slack通知失敗] {response.status_code} - {response.text}")


def _notify_telegram_impl(message: str):
    chat_id = _get_telegram_chat_id()
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # シンプルなテキスト送信（必要なら parse_mode='MarkdownV2' 等に）
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload, timeout=10)
    if response.status_code != 200:
        print(f"[Telegram通知失敗] {response.status_code} - {response.text}")


def notify_slack(message: str):
    """
    既存互換のエントリポイント。
    config.ini の settings:Setdefault に従って Slack または Telegram へ送信する。
    """
    global msg_history

    # 連投抑止（同一メッセージの短時間連投を避け、ログに記録）
    now = time.time()
    last_sent = _last_notify_times.get(message)
    msg_hash = hashlib.sha256(message.encode()).hexdigest()

    if msg_hash == msg_history or (last_sent and (now - last_sent < _NOTIFY_COOLDOWN_SECONDS)):
        log_message = f"[通知制限] メッセージ送信抑制: {message}（{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}）\n"
        try:
            with open("notification_log.txt", "a", encoding="utf-8") as f:
                f.write(log_message)
        except Exception:
            pass
        return
    _last_notify_times[message] = now

    # Debug モードならメッセージに印を付与
    if debug:
        message = "[Debug モード] " + message

    try:
        if str(default_service).lower() == "telegram":
            _notify_telegram_impl(message)
        else:
            # 既定は Slack
            _notify_slack_impl(message)
        msg_history = msg_hash
    except Exception as e:
        print(f"[通知例外] {e}")
