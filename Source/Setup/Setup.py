import mysql.connector

def create_database_and_tables():
    try:
        # DB接続情報
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='your_password'  # 必要に応じて変更
        )
        cursor = conn.cursor()

        # データベース作成
        cursor.execute("DROP DATABASE IF EXISTS Trade")
        cursor.execute("CREATE DATABASE Trade DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
        cursor.execute("USE Trade")

        # bot_config テーブル作成
        cursor.execute("""
        CREATE TABLE bot_config (
            `key` VARCHAR(64) NOT NULL,
            `value` VARCHAR(64) NOT NULL,
            PRIMARY KEY (`key`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
        """)

        # bot_config データ挿入
        cursor.executemany("""
        INSERT INTO bot_config (`key`, `value`) VALUES (%s, %s)
        """, [
            ('CHECK_INTERVAL', '3'),
            ('LOT_SIZE', '1000'),
            ('MACD_DIFF_THRESHOLD', '0.0015'),
            ('MAINTENANCE_MARGIN_RATIO', '0.5'),
            ('MAX_LOSS', '20'),
            ('MAX_SPREAD', '0.03'),
            ('MIN_PROFIT', '40'),
            ('SKIP_MODE', '0'),
            ('SYMBOL', 'USD_JPY'),
            ('TIME_STOP', '22'),
            ('VOL_THRESHOLD', '0.03')
        ])

        # trade_logs テーブル作成
        cursor.execute("""
        CREATE TABLE trade_logs (
            `id` INT NOT NULL AUTO_INCREMENT,
            `timestamp` DATETIME NOT NULL,
            `action` VARCHAR(255) NOT NULL,
            `price` DECIMAL(15,5) NOT NULL,
            PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
        """)

        # 必要に応じて trade_logs の初期データ挿入も可能（大量なので省略）

        conn.commit()
        print("データベースとテーブルの作成が完了しました。")

    except mysql.connector.Error as err:
        print(f"エラー: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    create_database_and_tables()
