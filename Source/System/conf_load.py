import os
import sqlite3
from pathlib import Path
import mysql.connector
from dotenv import load_dotenv

# .env読み込み
load_dotenv()

DB_PATH = Path("api_settings.db")

def load_apifile_conf():
    import configparser
    
    # 設定ファイル読み込み
    config = configparser.ConfigParser()
    config.read("/opt/Innovations/System/config.ini", encoding="utf-8")
    log_level = config.get("API", "SOURCE", fallback="file")# デフォルトは有効(1)
    return log_level

def load_settings_from_sqlite():
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name, value FROM api_settings")
            settings = {name: value for name, value in cursor.fetchall()}
            cursor.close()
            conn.close()
            # print("✅ SQLite (api_settings.db) から設定を読み込みました。")
            return settings
        except sqlite3.Error as err:
            print(f"[エラー] SQLite接続エラー: {err}")
            # SQLiteに失敗したらMySQLにフォールバック
            pass

def load_settings_from_db():
    if load_apifile_conf() == "file":
        return load_settings_from_sqlite()

    """DBからAPIキーなどの設定を読み込む（優先順位: SQLite → MySQL）"""
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
