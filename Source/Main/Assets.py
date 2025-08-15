import requests
import json
import hmac
import hashlib
import time
from datetime import datetime

def assets(apiKey,secretKey):
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    method    = 'GET'
    endPoint  = 'https://forex-api.coin.z.com/private'
    path      = '/v1/account/assets'

    text = timestamp + method + path
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

    headers = {
        "API-KEY": apiKey,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.get(endPoint + path, headers=headers)
    return json.loads(res.text)

def get_positionLossGain(apiKey,secretKey):
    timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
    method    = 'GET'
    endPoint  = 'https://forex-api.coin.z.com/private'
    path      = '/v1/account/assets'

    text = timestamp + method + path
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

    headers = {
        "API-KEY": apiKey,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.get(endPoint + path, headers=headers)
    res_json = json.loads(res.text)
    
    inz =int(float(res_json["data"]["positionLossGain"]))
    with open("positionLossGain.txt", "a", encoding="utf-8") as f:
        f.write(f"{inz}\n")
    return float(res_json["data"]["positionLossGain"])
    # return float(res_json["data"][0]["positionLossGain"])
