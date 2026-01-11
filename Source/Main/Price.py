import logging

# 価格情報をレスポンスから抽出する関数
def extract_price_from_response(res):
    try:
        data = res.json()
        if isinstance(data, dict):
            dlist = data.get("data", [])
            if isinstance(dlist, list) and len(dlist) > 0:
                return dlist[0].get("price", "取得不可")
        return "データなし"
    except Exception as e:
        logging.warning(f"[レスポンス解析失敗] {e}")
        return "解析失敗"
