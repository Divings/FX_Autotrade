def load_apifile_conf():
    import configparser
    
    # 設定ファイル読み込み
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    log_level = config.get("API", "SOURCE", fallback="file")# デフォルトは有効(1)
    return log_level
print(load_apifile_conf())
input(" >>")