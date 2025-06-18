# slack_notify.py
import os
import requests
import time
from dotenv import load_dotenv
from conf_load import load_settings_from_db

import configparser

def load_config():
    # ConfigParser オブジェクトを作成
    config = configparser.ConfigParser()
    try:
        # config.ini を読み込む
        config.read('config.ini')
        debug = config.getboolean('settings', 'debug')
    except:
        debug = False
    return debug

debug=load_config()
# 設定をDBから読み込む
config1 = load_settings_from_db()
SLACK_WEBHOOK_URL = config1["SLACK_WEBHOOK_URL"]

# 通知制限管理辞書（メッセージ: 最後に送信した時刻）
_last_notify_times = {}
_NOTIFY_COOLDOWN_SECONDS = 60  # 同じ内容は60秒間送らない

def notify_slack(message: str):
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL is not set.")
        
    now = time.time()
    last_sent = _last_notify_times.get(message)
    if last_sent and (now - last_sent < _NOTIFY_COOLDOWN_SECONDS):
        # 通知制限中
        print(f"[Slack通知制限] メッセージ送信抑制: {message}")
        return
    _last_notify_times[message] = now

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
    if debug==True:
        message= "[Debug モード] " + message
    try:
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
