#!/bin/bash
SERVICE_NAME="fx-autotrade.service"
LIVE_DIR="/home/opc/autofx"
GIT_DIR="/home/opc/FX_Autotrade"
SOURCE_SUBDIR="Source/Main"

# GitHubの最新コードをpull
cd "$GIT_DIR" || exit 1
git pull origin main

# 差分チェック：Source/Mainとliveを比較
diff -r --exclude='.env' --exclude='__pycache__' --exclude='*.pyc'  --exclude='*.txt' --exclude='*.csv' --exclude='*.json'  --exclude='*.pkl' "$GIT_DIR/$SOURCE_SUBDIR" "$LIVE_DIR" > /tmp/diff_output.txt

if [ -s /tmp/diff_output.txt ]; then
    echo "差分あり、更新を開始します。"

    # サービス停止
    sudo systemctl stop "$SERVICE_NAME"

    # バックアップ（任意）
    # cp -r "$LIVE_DIR" "${LIVE_DIR}_backup_$(date +%Y%m%d_%H%M%S)"

    # 上書き（Source/Main→live）
    rsync -av --delete --exclude='.env' --exclude='__pycache__/**' --exclude='*.csv' --exclude='*.json'  --exclude='*.pkl' "$GIT_DIR/$SOURCE_SUBDIR/" "$LIVE_DIR/"
    
    # サービス再起動
    sudo systemctl start "$SERVICE_NAME"

    echo "更新完了、サービスを再起動しました。"
    sudo systemctl status "$SERVICE_NAME"
    python3 /home/opc/script/Update-Mmsg.py
else
    echo "差分なし、更新は不要です。"
fi
sync
