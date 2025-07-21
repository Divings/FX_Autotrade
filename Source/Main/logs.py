import os
import sqlite3
from pathlib import Path
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

# .envを読み込む
load_dotenv()

SETTINGS_DB = Path("api_settings.db")
LOG_DB = Path("trade_logs.db")

def write_log(action, price):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if SETTINGS_DB.exists():
        try:
            conn = sqlite3.connect(LOG_DB)
            cursor = conn.cursor()

            # trade_logs テーブルを作成（存在しない場合）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    action VARCHAR(255) NOT NULL,
                    price DECIMAL(15,5) NOT NULL
                )
            """)

            sql = "INSERT INTO trade_logs (timestamp, action, price) VALUES (?, ?, ?)"
            cursor.execute(sql, (timestamp, action, price))
            conn.commit()

            # print("✅ ログを SQLite (trade_logs.db) に書き込みました。")

        except sqlite3.Error as err:
            print(f"[ERROR] SQLite エラー: {err}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        return

    # SQLite が無ければ MySQL に書き込む
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor()

        sql = "INSERT INTO trade_logs (timestamp, action, price) VALUES (%s, %s, %s)"
        values = (timestamp, action, price)

        cursor.execute(sql, values)
        conn.commit()

        # print("✅ ログを MySQL に書き込みました。")

    except mysql.connector.Error as err:
        print(f"[ERROR] MySQL エラー: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
