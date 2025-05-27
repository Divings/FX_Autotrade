# slack_notify.py
import os
import requests
from dotenv import load_dotenv
from conf_load import load_settings_from_db
config1=load_settings_from_db()
SLACK_WEBHOOK_URL = config1["SLACK_WEBHOOK_URL"]

def notify_slack(message: str):
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL is not set.")
    
    payload = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Slack通知失敗: {response.status_code} - {response.text}")