import hashlib
import json
from pathlib import Path
import shutil
import os
Main_dir=os.path.dirname(os.path.abspath(__file__))

def calculate_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

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
        print(f"Copied: {rel_name}")
        recorded_hashes[rel_name] = file_hash
    else:
        print(f"Skipped (unchanged): {rel_name}")

# ハッシュファイル更新
with open(hash_file, 'w') as f:
    json.dump(recorded_hashes, f, indent=2)
import subprocess
v=input(" リポジトリにプッシュしますか？ (Y or N)>> ")
if v.lower()=="y":
    message=input(" Commit Messagre >> ")
    subprocess.run(f"git commit -m \"{message}\"")
    subprocess.run("git push git@github.com:Divings/FX_Autotrade.git")
    print(" コードをプッシュしました")
input(" >> ")