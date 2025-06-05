import asyncio
from dotenv import load_dotenv
import os
import socket
# 認証トークン（適宜設定）
load_dotenv()  # .envを読み込み

VALID_TOKEN = os.getenv("VALID_TOKEN", "default-token")

async def handle_client(reader, writer,shared_state=None):
    try:
        data = await reader.read(100)
        message = data.decode().strip()
        addr = writer.get_extra_info('peername')

        if message == VALID_TOKEN:
            shared_state["cmd"] = "save_adx"  # ★この行が重要！
            response = "[OK] 保存コマンドを受け付けました\n"
        else:
            response = "[ERROR] 認証失敗\n"

        writer.write(response.encode())
        await writer.drain()
        await asyncio.sleep(0.1)
        writer.close()
    except Exception as e:
        print(f"[エラー] handle_client: {e}")

async def start_socket_server(shared_state, host='127.0.0.1', port=8888):
    async def handler(reader, writer):
        await handle_client(reader, writer, shared_state)

    server = await asyncio.start_server(handler, host, port)
    print(f"[待機中] ソケットサーバー {host}:{port}")
    async with server:
        await server.serve_forever()
