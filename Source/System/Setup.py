import sqlite3
from pathlib import Path
import requests
import base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

DB_PATH = Path("api_settings.db")
KEY_FILE = Path("aes_key.bin")

# =========================
# AESキー管理
# =========================
def load_or_create_aes_key():
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()

    key = get_random_bytes(32)  # 256bit
    KEY_FILE.write_bytes(key)
    try:
        KEY_FILE.chmod(0o600)
    except:
        pass
    return key

AES_KEY = load_or_create_aes_key()

# =========================
# AES-GCM 暗号化 / 復号
# =========================
def aes_encrypt(text: str) -> str:
    cipher = AES.new(AES_KEY, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(text.encode())
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

def aes_decrypt(token: str) -> str:
    raw = base64.b64decode(token)
    nonce = raw[:16]
    tag = raw[16:32]
    ciphertext = raw[32:]
    cipher = AES.new(AES_KEY, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode()

# =========================
# DB セットアップ
# =========================
def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL
        )
    """)
    conn.commit()

    api_key = input("🔷 API_KEY を入力してください: ").strip()
    api_secret = input("🔷 API_SECRET を入力してください: ").strip()
    slack_webhook = input("🔷 SLACK_WEBHOOK_URL を入力してください: ").strip()

    # 🔐 AES暗号化
    api_key_enc = aes_encrypt(api_key)
    api_secret_enc = aes_encrypt(api_secret)
    slack_webhook_enc = aes_encrypt(slack_webhook)

    cursor.execute(
        "INSERT OR REPLACE INTO api_settings (name, value) VALUES (?, ?)",
        ("API_KEY", api_key_enc)
    )
    cursor.execute(
        "INSERT OR REPLACE INTO api_settings (name, value) VALUES (?, ?)",
        ("API_SECRET", api_secret_enc)
    )
    cursor.execute(
        "INSERT OR REPLACE INTO api_settings (name, value) VALUES (?, ?)",
        ("SLACK_WEBHOOK_URL", slack_webhook_enc)
    )

    url_value = "https://github.com/Divings/Public_Auto_Trade_pac/releases/download/Pubkey/"
    cursor.execute(
        "INSERT OR REPLACE INTO api_settings (name, value) VALUES (?, ?)",
        ("URL", url_value)
    )

    conn.commit()
    conn.close()
    print(f"\n🎉 セットアップ完了（AES暗号化済）: {DB_PATH}")

# =========================
# 実行部
# =========================
if __name__ == "__main__":

    url = "https://github.com/Divings/Public_Auto_Trade_pac/releases/download/bot_config/bot_config.xml"
    save_path = "/opt/Innovations/System/bot_config.xml"

    try:
        response = requests.get(url)
        response.raise_for_status()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"設定テンプレート取得: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"ダウンロードエラー: {e}")

    if DB_PATH.exists():
        overwrite = input(f"⚠ 既に {DB_PATH} が存在します。上書きしますか？ (y/N): ").strip().lower()
        if overwrite != "y":
            print("🚫 キャンセルしました。")
            exit(0)
        DB_PATH.unlink()
        print("🗑 古いDBを削除しました")

    setup_database()
