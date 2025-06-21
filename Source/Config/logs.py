import os
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

# .envを読み込む
load_dotenv()

def write_log(action, price):
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor()

        sql = "INSERT INTO trade_logs (timestamp, action, price) VALUES (%s, %s, %s)"
        values = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action, price)

        cursor.execute(sql, values)
        conn.commit()

    except mysql.connector.Error as err:
        print(f"[ERROR] MySQLエラー: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
