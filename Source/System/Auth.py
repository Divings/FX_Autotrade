import tempfile,requests
from slack_notify import notify_slack
import sys
import subprocess
import os

temp_dir = tempfile.mkdtemp()
# パラメータ設定
PUBLIC_KEY_URL =  "https://github.com/Divings/Public_Auto_Trade_pac/releases/download/Pubkey/publickey.asc"
PUBLIC_KEY_FILE = "/opt/gpg/publickey.asc"
UPDATE_FILE = "AutoTrade.py"
SIGNATURE_FILE = "AutoTrade.py.sig"

# 設定ファイル読み込み関数
def load_conf():
    import configparser
    
    # 設定ファイル読み込み
    config = configparser.ConfigParser()
    config.read("/opt/Innovations/System/config.ini", encoding="utf-8")
    log_level = config.getint("Auth", "enable", fallback=1)# デフォルトは有効(1)
    return log_level

def download_public_key(url, save_path):
    """公開鍵をダウンロードして保存"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        # print("公開鍵をダウンロードしました")
    except Exception as e:
        #notify_slack(f"公開鍵ダウンロード失敗: {str(e)}")
        sys.exit(1)

def import_public_key(gpg_home, key_path):
    """公開鍵をGPGにインポート"""
    try:
        subprocess.run(['gpg', '--homedir', gpg_home, '--import', key_path], check=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print("公開鍵をインポートしました")
    except subprocess.CalledProcessError:
        # notify_slack("公開鍵インポート失敗")
        sys.exit(1)

def verify_signature(gpg_home, signature_file, update_file):
    """署名検証"""
    result = subprocess.run(
        ['gpg', '--homedir', gpg_home, '--verify', signature_file, update_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        notify_slack(f"署名検証失敗！起動中止:{update_file}")
        print(result.stdout)
        print(result.stderr)
        return 1
    # notify_slack("[INFO] 署名検証成功")
    return 0

# 設定ファイルを読み込み
mode=load_conf() # 1:有効,0:無効

# モードが無効なら終了(正常起動とする)
if mode == 0:
    notify_slack("[INFO] 署名検証が無効です。\n処理のカスタマイズが可能です")
    sys.exit(0)

# 公開鍵ダウンロード・インポート・署名検証
public_key_path = os.path.join(temp_dir, "publickey.asc")
download_public_key(PUBLIC_KEY_URL, public_key_path)

# 公開鍵インポート
import_public_key(temp_dir, public_key_path)

# 署名検証
out = verify_signature(temp_dir, SIGNATURE_FILE, UPDATE_FILE)
# if out == 0:
#    notify_slack("[INFO] 署名検証成功")

# 終了コードを返す
sys.exit(out)