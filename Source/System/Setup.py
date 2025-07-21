import sqlite3
from pathlib import Path

DB_PATH = Path("api_settings.db")

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # テーブル作成
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL
        )
    """)
    conn.commit()

    # 対話式で API_KEY と API_SECRET と SLACK_WEBHOOK_URL を入力
    api_key = input("🔷 API_KEY を入力してください: ").strip()
    api_secret = input("🔷 API_SECRET を入力してください: ").strip()
    slack_webhook = input("🔷 SLACK_WEBHOOK_URL を入力してください: ").strip()

    cursor.execute("""
        INSERT OR REPLACE INTO api_settings (name, value) VALUES (?, ?)
    """, ("API_KEY", api_key))
    cursor.execute("""
        INSERT OR REPLACE INTO api_settings (name, value) VALUES (?, ?)
    """, ("API_SECRET", api_secret))
    cursor.execute("""
        INSERT OR REPLACE INTO api_settings (name, value) VALUES (?, ?)
    """, ("SLACK_WEBHOOK_URL", slack_webhook))

    # 固定で URL を追加（新しいURL）
    url_value = "https://github.com/Divings/FX_Autotrade/releases/download/Pubkey/"
    cursor.execute("""
        INSERT OR REPLACE INTO api_settings (name, value) VALUES (?, ?)
    """, ("URL", url_value))

    conn.commit()
    conn.close()
    print(f"\n🎉 セットアップ完了: {DB_PATH}")

if __name__ == "__main__":
    if DB_PATH.exists():
        overwrite = input(f"⚠ 既に {DB_PATH} が存在します。上書きしますか？ (y/N): ").strip().lower()
        if overwrite != "y":
            print("🚫 キャンセルしました。")
            exit(0)
        DB_PATH.unlink()
        print("🗑 古いデータベースを削除しました。")

    setup_database()
