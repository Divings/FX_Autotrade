import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os

# .env を読み込む
load_dotenv()

# 環境変数からDB情報を取得
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

def insert_data(table: str, columns: list, values: tuple) -> bool:
    """
    指定したテーブルにデータを1行挿入する関数

    Args:
        table (str): テーブル名
        columns (list): 挿入するカラムのリスト
        values (tuple): 挿入する値のタプル

    Returns:
        bool: 成功したらTrue、失敗したらFalse
    """
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        if conn.is_connected():
            cursor = conn.cursor()
            cols_str = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(values))
            sql = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders})"
            cursor.execute(sql, values)
            conn.commit()
            #print(f"🎉 {cursor.rowcount} 件追加しました")
            return True

    except Error as e:
        #print(f"❌ エラーが発生しました: {e}")
        return False

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            #print("🔒 接続を閉じました")
