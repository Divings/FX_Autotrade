import pandas as pd
import matplotlib.pyplot as plt
import os

# === CSVファイル読み込み ===
LOG_FILE = "fx_trade_log.csv"

if not os.path.exists(LOG_FILE):
    print("取引ログファイルが存在しません。")
    exit(1)


# CSV読み込み
df = pd.read_csv(LOG_FILE)

# timestampをdatetime型に変換
df["timestamp"] = pd.to_datetime(df["timestamp"])
# CSV読み込み
df = pd.read_csv(LOG_FILE)

# timestampをdatetime型に変換
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 損益列がなければ新規作成
profits = []
buy_price = None

for _, row in df.iterrows():
    if row["action"] == "BUY":
        buy_price = row["price"]
        profits.append(0.0)  # エントリー時点では損益なし
    elif row["action"] in ("SELL", "LOSS_CUT") and buy_price is not None:
        profit = row["price"] - buy_price
        profits.append(profit)
        buy_price = None  # ポジションを解消
    else:
        profits.append(0.0)

df["profit"] = profits

# 売却（SELL, LOSS_CUT）と購入（BUY）を分ける
buys = df[df["action"] == "BUY"].copy()
sells = df[df["action"].isin(["SELL", "LOSS_CUT"])]

# 損益列があればfloat型に変換（なければ0）
try:
    df["profit"] = pd.to_numeric(df.get("profit", 0), errors="coerce").fillna(0.0)
except:
    df["profit"]=0
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
# 損益の推移グラフ
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

# エントリーと決済価格の履歴可視化（任意）
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

