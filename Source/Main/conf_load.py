import os
import mysql.connector
from dotenv import load_dotenv

# .env読み込み
load_dotenv()

def load_settings_from_db():
    """MySQLからAPIキーなどの設定を読み込む"""
    try:
        # 接続設定を.envから取得
        db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME')
        }

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 設定読み込み
        cursor.execute("SELECT name, value FROM api_settings")
        settings = {name: value for name, value in cursor.fetchall()}

        cursor.close()
        conn.close()

        return settings

    except mysql.connector.Error as err:
        print(f"[エラー] MySQL接続エラー: {err}")
        return {}