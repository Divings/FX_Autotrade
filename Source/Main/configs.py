import configparser
import os

def load_weekconfigs():
    if os.path.exists('config.ini'):
        # ConfigParser オブジェクトを作成
        config = configparser.ConfigParser()

        # ファイルを読み込む
        config.read('config.ini')

        # 値を取得
        host = config.get('settings', 'value')
        return host
    else:
        return 0
