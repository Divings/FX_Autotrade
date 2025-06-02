# slack_notify.py
import os
import requests
from dotenv import load_dotenv
from conf_load import load_settings_from_db

# 設定をDBから読み込む
config1 = load_settings_from_db()
SLACK_WEBHOOK_URL = config1["SLACK_WEBHOOK_URL"]

def notify_slack(message: str):
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL is not set.")

    # メッセージの内容に応じて色を設定
    if "[即時損切]" in message or "[決済] 損切り" in message or "[\u26a0\ufe0fアラート]" in message:
        color = "#ff4d4d"  # 赤
    elif "[決済]" in message or "[即時利確]" in message or "[RSI" in message:
        color = "#36a64f"  # 緑
    elif "[保有]" in message:
        color = "#439FE0"  # 青
    elif "[建玉]" in message or "[スプレッド]" in message:
        color = "#daa520"  # オレンジ
    elif "[エラー]" in message or "[注意]" in message:
        color = "#8b0000"  # 暗赤
    elif "[INFO]" in message:
        color = "#888888"  # グレー（情報）
    else:
        color = "#dddddd"  # 既定（薄グレー）
    try:
        # Slackに通知
        payload = {
            "attachments": [
                {
                    "color": color,
                    "text": message
                }
            ]
        }

        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code != 200:
            print(f"[Slack通知失敗] {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[Slack通知例外] {e}")
