import os
import warnings
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mysql.connector
from dotenv import load_dotenv

# 警告非表示
warnings.filterwarnings("ignore")

# === 日本語フォント設定（Windows用） ===
font_path = "C:/Windows/Fonts/meiryo.ttc"
plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()

# === .env読み込み ===
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_TABLE = os.getenv("DB_TABLE", "trade_logs")

# === DB接続・データ取得 ===
try:
    conn = mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME
    )
    df = pd.read_sql(f"SELECT timestamp, action, price FROM {DB_TABLE} ORDER BY timestamp ASC", conn)
    conn.close()
except Exception as e:
    print("データベース接続エラー:", e)
    exit(1)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# === 損益計算 ===
profits = []
buy_price = None
for _, row in df.iterrows():
    action = row["action"]
    price = row["price"]
    if action == "BUY":
        buy_price = price
        profits.append(0.0)
    elif action in ("SELL", "LOSS_CUT") and buy_price is not None:
        profits.append(price - buy_price)
        buy_price = None
    else:
        profits.append(0.0)

df["profit"] = profits
df["cumulative_profit"] = df["profit"].cumsum()

# === 成績統計 ===
closed_trades = df[df["action"].isin(["SELL", "LOSS_CUT"])]
total_profit = closed_trades["profit"].sum()
wins = (closed_trades["profit"] > 0).sum()
losses = (closed_trades["profit"] <= 0).sum()
total_trades = wins + losses
win_rate = (wins / total_trades * 100) if total_trades else 0.0

print(f"総損益: {total_profit:.2f} 円")
print(f"取引回数: {total_trades} 回 (勝ち: {wins}, 負け: {losses})")
print(f"勝率: {win_rate:.2f} %")

# === グラフ生成 ===
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["cumulative_profit"], marker="o")
plt.title("累積損益の推移")
plt.xlabel("日時")
plt.ylabel("累積損益 (円)")
plt.grid(True)
plt.tight_layout()
plt.savefig("profit_chart.png")
plt.show()

# === 売買履歴可視化 ===
buys = df[df["action"] == "BUY"]
sells = df[df["action"].isin(["SELL", "LOSS_CUT"])]
if not buys.empty and not sells.empty:
    plt.figure(figsize=(10, 5))
    plt.plot(buys["timestamp"], buys["price"], "go", label="BUY")
    plt.plot(sells["timestamp"], sells["price"], "rx", label="SELL / LOSS_CUT")
    plt.title("エントリー・決済価格履歴")
    plt.xlabel("日時")
    plt.ylabel("価格 (円)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("price_history_chart.png")
    plt.show()

input(" >> ")
