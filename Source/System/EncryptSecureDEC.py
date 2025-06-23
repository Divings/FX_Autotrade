import os
import getpass
import sys
import lzma
import hashlib
import json
from datetime import datetime, timezone
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2

BLOCKCHAIN_HEADER = b'BLOCKCHAIN_DATA_START\n'

class Block:
    def __init__(self, data, previous_hash, operation_type, file_hash, user, memo):
        self.timestamp = datetime.now(timezone.utc)
        self.data = data
        self.previous_hash = previous_hash
        self.operation_type = operation_type
        self.file_hash = file_hash
        self.user = user
        self.memo = memo
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        sha = hashlib.sha256()
        sha.update(
            str(self.timestamp).encode('utf-8') +
            str(self.data).encode('utf-8') +
            str(self.previous_hash).encode('utf-8') +
            str(self.operation_type).encode('utf-8') +
            str(self.file_hash).encode('utf-8') +
            str(self.user).encode('utf-8') +
            str(self.memo).encode('utf-8')
        )
        return sha.hexdigest()

    def to_dict(self):
        return {
            'timestamp': str(self.timestamp),
            'data': self.data,
            'previous_hash': self.previous_hash,
            'operation_type': self.operation_type,
            'file_hash': self.file_hash,
            'user': self.user,
            'memo': self.memo,
            'hash': self.hash
        }

class Blockchain:
    def __init__(self):
        self.chain = []

    def add_block(self, new_block):
        if len(self.chain) == 0:
            new_block.previous_hash = "0"
        else:
            new_block.previous_hash = self.chain[-1].hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def to_json(self):
        return json.dumps([block.to_dict() for block in self.chain], indent=2)

    @staticmethod
    def from_json(data):
        chain_data = json.loads(data)
        blockchain = Blockchain()
        for block_data in chain_data:
            block = Block(
                data=block_data['data'],
                previous_hash=block_data['previous_hash'],
                operation_type=block_data['operation_type'],
                file_hash=block_data['file_hash'],
                user=block_data['user'],
                memo=block_data['memo']
            )
            block.timestamp = datetime.strptime(block_data['timestamp'], '%Y-%m-%d %H:%M:%S.%f%z')
            block.hash = block_data['hash']
            blockchain.chain.append(block)
        return blockchain

def decrypt_file(file_path, password, memo=""):
    with lzma.open(file_path, 'rb') as f:
        data = f.read()

    split_index = data.index(BLOCKCHAIN_HEADER)
    crypto_data = data[:split_index]
    chain_json = data[split_index + len(BLOCKCHAIN_HEADER):].decode('utf-8')
    blockchain = Blockchain.from_json(chain_json)

    salt = crypto_data[:16]
    nonce = crypto_data[16:28]
    tag = crypto_data[-16:]
    ciphertext = crypto_data[28:-16]

    key = PBKDF2(password, salt, dkLen=32, count=100_000)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    try:
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError:
        raise ValueError("Decryption failed. The password may be incorrect or the file may be tampered with.")

    output_file = file_path.replace(".vdec", "_decrypted")
    with open(output_file, 'wb') as f:
        f.write(plaintext)

    username = getpass.getuser()
    file_hash = hashlib.sha256(ciphertext).hexdigest()
    block = Block(file_hash, blockchain.chain[-1].hash if blockchain.chain else "0", "Decrypt", file_hash, username, memo)
    blockchain.add_block(block)

    with lzma.open(file_path, 'wb') as f:
        f.write(salt + nonce + ciphertext + tag)
        f.write(BLOCKCHAIN_HEADER)
        f.write(blockchain.to_json().encode('utf-8'))

    return output_file

