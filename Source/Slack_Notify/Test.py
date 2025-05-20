# test_slack_notify.py

from slack_notify import notify_slack

if __name__ == "__main__":
    try:
        notify_slack("Slack通知テスト成功！このメッセージが届けば設定完了です。")
        print("Slack通知送信成功")
    except Exception as e:
        print(f"Slack通知送信失敗: {e}")