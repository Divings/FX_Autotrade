
def write_indo(id):
    endPoint  = 'https://forex-api.coin.z.com/private'
    path      = '/v1/orders'
    method    = 'GET'
    timestamp = str(int(time.time() * 1000))  # より精度の高いミリ秒
    # sign = create_signature(timestamp, method, path, reqBody)

    text = timestamp + method + path
    sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()
    parameters = { "rootOrderId": id}

    headers = {
        "API-KEY": apiKey,
        "API-TIMESTAMP": timestamp,
        "API-SIGN": sign
    }

    res = requests.get(endPoint + path, headers=headers, params=parameters)
    print (json.dumps(res.json(), indent=2))