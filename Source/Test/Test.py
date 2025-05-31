from slack_notify import notify_slack
def mentnancemode(shared_state):
    while True:
        if shared_state["last_skip_notice"]==False:
            notify_slack(f"現在メンテナンスモード中です。")
            shared_state["last_skip_notice"]=True
        else:
            shared_state["last_skip_notice"]=False