import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
import warnings
import mysql.connector
from dotenv import load_dotenv
import datetime

# 警告を非表示
warnings.filterwarnings("ignore")

# === 日本語フォント設定（Windows用） ===
font_path = "C:/Windows/Fonts/meiryo.ttc"
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

# .envからDB情報を読み込み
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_TABLE = os.getenv("DB_TABLE", "trade_logs")

# MySQLからデータ取得
try:
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
except Exception as e:
    print("データベース接続エラー:", e)
    exit(1)

query = f"SELECT timestamp, action, price FROM {DB_TABLE} ORDER BY timestamp ASC"
df = pd.read_sql(query, conn)
conn.close()

# timestampをdatetime型に変換
df["timestamp"] = pd.to_datetime(df["timestamp"])

# === 損益列の計算 ===
profits = []
buy_price = None

for _, row in df.iterrows():
    if row["action"] == "BUY":
        buy_price = row["price"]
        profits.append(0.0)
    elif row["action"] in ("SELL", "LOSS_CUT") and buy_price is not None:
        profit = row["price"] - buy_price
        profits.append(profit)
        buy_price = None
    else:
        profits.append(0.0)

df["profit"] = profits

# 売却（SELL, LOSS_CUT）と購入（BUY）を分ける
buys = df[df["action"] == "BUY"].copy()
sells = df[df["action"].isin(["SELL", "LOSS_CUT"])]

# 総損益・勝率算出
profits = df[df["action"].isin(["SELL", "LOSS_CUT"])]
total_profit = profits["profit"].sum()
wins = profits[profits["profit"] > 0].shape[0]
losses = profits[profits["profit"] <= 0].shape[0]
total = wins + losses
win_rate = (wins / total * 100) if total > 0 else 0.0

print(f"総損益: {total_profit:.2f} 円")
print(f"取引回数: {total} 回 (勝ち: {wins}, 負け: {losses})")
print(f"勝率: {win_rate:.2f} %")

# === グラフ化 ===
# 累積損益グラフ
df["cumulative_profit"] = df["profit"].cumsum()
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["cumulative_profit"], marker="o")
plt.title("累積損益の推移")
plt.xlabel("日時")
plt.ylabel("累積損益 (円)")
plt.grid(True)
plt.tight_layout()
plt.savefig("profit_chart.png")
plt.show()

# エントリーと決済価格の履歴可視化
if not buys.empty and not sells.empty:
    plt.figure(figsize=(10, 5))
    plt.plot(buys["timestamp"], buys["price"], "go", label="BUY")
    plt.plot(sells["timestamp"], sells["price"], "rx", label="SELL/LOSS_CUT")
    plt.title("エントリー・決済価格履歴")
    plt.xlabel("日時")
    plt.ylabel("価格 (円)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("price_history_chart.png")
    plt.show()
