import hashlib
import json
from pathlib import Path
import shutil
import os
import sys

print("")
Main_dir=os.path.dirname(os.path.abspath(__file__))
print("")
def calculate_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

print("")

# このスクリプトのパス
script_dir = Path(__file__).resolve().parent

# Main フォルダ
main_dir = script_dir.parent

# System フォルダ（Main の1つ上）
system_dir = main_dir.parent / "System"
system_dir.mkdir(exist_ok=True)

# ハッシュ記録ファイル
hash_file = script_dir / "hashes.json"
if hash_file.exists():
    with open(hash_file, 'r') as f:
        recorded_hashes = json.load(f)
else:
    recorded_hashes = {}

# Main直下の.pyファイルだけを対象（Scriptなどサブディレクトリは除外）
for py_file in main_dir.glob("*.py"):
    file_hash = calculate_hash(py_file)
    rel_name = py_file.name
    target_file = system_dir / rel_name

    if recorded_hashes.get(rel_name) != file_hash:
        shutil.copy2(py_file, target_file)
        print(f" Copied: {rel_name}")
        recorded_hashes[rel_name] = file_hash
    else:
        print(f" Skipped (unchanged): {rel_name}")

# ハッシュファイル更新
with open(hash_file, 'w') as f:
    json.dump(recorded_hashes, f, indent=2)
import subprocess

v=input(" リポジトリにプッシュしますか？ (Y or N)>> ")
if v.lower()=="y":
    print("")
    message=input(" Commit Messagre >> ")
    subprocess.run("git add *")
    subprocess.run(f"git commit -a -m \"{message}\"")
    subprocess.run("git push https://github.com/Divings/FX_Autotrade.git")
    print(" コードをプッシュしました")
    v=1
else:
    print(" コードのプッシュをスキップしました\n 手動でプッシュしてください")
    v=0

if v==0 or v==1:
    sys.exit()

import mysql.connector
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os

# .env 読み込み
load_dotenv(dotenv_path=script_dir / '.env')  # script_dir は既に定義済み

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_name = os.getenv("DB_NAME")

# AutoTrade.py のパス
autotrade_path = main_dir / "AutoTrade.py"

# SYS_VER を読み取る関数
def read_sys_ver(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r"\s*SYS_VER\s*=\s*['\"](.*?)['\"]", line)
            if m:
                return m.group(1)
    return None

sys_ver = read_sys_ver(autotrade_path)

if sys_ver:
    print(f"AutoTrade.py SYS_VER: {sys_ver}")

    # MySQL に書き込む
    try:
        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_pass,
            database=db_name
        )

        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sys_ver_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                version VARCHAR(255) NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """)
        c.execute("""
            INSERT INTO sys_ver_log (version, timestamp) VALUES (%s, %s)
        """, (sys_ver, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        print("SYS_VER を MySQL に記録しました。")

    except mysql.connector.Error as err:
        print(f"MySQL エラー: {err}")
else:
    print("SYS_VER が AutoTrade.py から見つかりませんでした。")
