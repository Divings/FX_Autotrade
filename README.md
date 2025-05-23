# AutoTrade.py - GMOコインFX 自動売買ボット

このリポジトリは、GMOコインの外国為替（FX）APIを用いた自動売買プログラムを提供します。主に USD/JPY を対象とし、移動平均線を使ったトレンド判定および損益基準による自動売買を行います。

## 📁 フォルダ構成

```
Source
├── Backup        # バックアップコード（旧バージョンなど）
├── Main          # メインのトレードロジック（AutoTrade.py）
└── Util          # 補助スクリプト（署名ツール・設定補助など）
```

## 🚀 特徴

- GMOコインFX（FOREX）API対応
- USD/JPY 固定の通貨ペア
- ローリング移動平均によるトレンド検出
- 自動エントリー・利確・損切り
- 証拠金維持率チェックとアラート
- ログ・CSV出力によるトレード履歴の記録

## 🔧 必要な環境

- Python 3.8 以上
- `.env` ファイルに APIキーの設定が必要
- Windows/Linux/macOS 対応

### インストール方法

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/Source/Main
pip install -r requirements.txt
```

### `.env` ファイルの例

```env
API_KEY=あなたのAPIキー
API_SECRET=あなたのAPIシークレット
```

## 📈 ロジック概要

### 売買条件
- 保有ポジションがない場合：
  - 移動平均差で `BUY` or `SELL` を判断
  - トレンドが不明な場合は2回連続でBUYを強行
- 保有ポジションがある場合：
  - 利益が40円以上 → 利確
  - 損失が20円以上 → 損切り

### トレンド判定
- 短期：6期間
- 長期：13期間
- サンプリング間隔：5秒ごと2分間

## 📁 出力ファイル

- `fx_debug_log.txt`: ログ出力
- `fx_trade_log.csv`: 売買履歴（タイムスタンプ、アクション、価格）

## ⚠️ 注意事項

- 本ツールは実際の資金を使った売買を行います。必ず十分なテストと理解を持ってご使用ください。
- テスト用途ではデモAPIがないため、実環境APIキーを使用する必要があります。
- 実行中は Ctrl+C で停止してください。

## 📄 ライセンス
This software is currently provided under a proprietary license. The previous version was under the MIT license, but all versions after April 12, 2025, will have restrictions on commercial use, redistribution, and modification.