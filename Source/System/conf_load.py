import os
import sqlite3
from pathlib import Path
import mysql.connector
from dotenv import load_dotenv

# .env読み込み
load_dotenv()

DB_PATH = Path("api_settings.db")

def load_settings_from_db():
    """DBからAPIキーなどの設定を読み込む（優先順位: SQLite → MySQL）"""
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name, value FROM api_settings")
            settings = {name: value for name, value in cursor.fetchall()}
            cursor.close()
            conn.close()
            print("✅ SQLite (api_settings.db) から設定を読み込みました。")
            return settings
        except sqlite3.Error as err:
            print(f"[エラー] SQLite接続エラー: {err}")
            # SQLiteに失敗したらMySQLにフォールバック
            pass

    print("⚠ SQLiteが無いかエラーのため、MySQLから設定を読み込みます。")
    try:
        # 接続設定を.envから取得
        db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASS'),
            'database': os.getenv('DB_NAME')
        }

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("SELECT name, value FROM api_settings")
        settings = {name: value for name, value in cursor.fetchall()}

        cursor.close()
        conn.close()

        return settings

    except mysql.connector.Error as err:
        print(f"[エラー] MySQL接続エラー: {err}")
        return {}
