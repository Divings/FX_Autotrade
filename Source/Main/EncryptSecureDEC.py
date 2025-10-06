# Copyright (c) 2025 Innovation Craft Inc. All Rights Reserved.
# 本ソフトウェアはプロプライエタリライセンスに基づき提供されています。

import argparse
import os
import getpass
import sys
import lzma
import hashlib
import json
import datetime
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
import rsa_signer
import wavencode
import rsa_encryptor

BLOCKCHAIN_HEADER = b'BLOCKCHAIN_DATA_START\n'

class Block:
    def __init__(self, data, previous_hash, operation_type, file_hash, user, memo):
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)
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
            block.timestamp = datetime.datetime.strptime(block_data['timestamp'], '%Y-%m-%d %H:%M:%S.%f%z')
            block.hash = block_data['hash']
            blockchain.chain.append(block)
        return blockchain

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.previous_hash != previous.hash:
                return False
            if current.calculate_hash() != current.hash:
                return False
        return True

deletemode = {
    "mode": False
}

def delete_pre_file(file_path):
    from pathlib import Path
    path = Path(file_path)
    abs_path = path.resolve()
    if abs_path.exists():
        abs_path.unlink()

def cli_encrypt(file_path, password, memo):
    import passchk 
    with open(file_path, 'rb') as f:
        plaintext = f.read()
    salt = get_random_bytes(16)
    
    if passchk.passchk(memo, password):
        print(" Warning: The note contains a password.\n For security reasons, it is recommended not to include passwords in notes.")
        a = input(" Do you want to continue (y or n) >> ")
        if a.lower() != "y":
            print(" Canceled the encryption.")
            return 
        
    key = PBKDF2(password, salt, dkLen=32, count=100_000)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    file_hash = hashlib.sha256(ciphertext).hexdigest()
    username = getpass.getuser()

    try:
        with lzma.open(file_path + ".vdec", 'rb') as f:
            data = f.read()
        split_index = data.index(BLOCKCHAIN_HEADER)
        chain_json = data[split_index + len(BLOCKCHAIN_HEADER):].decode('utf-8')
        blockchain = Blockchain.from_json(chain_json)
    except:
        blockchain = Blockchain()
    block = Block(file_hash, blockchain.chain[-1].hash if blockchain.chain else "0", "Encrypt", file_hash, username, memo)
    blockchain.add_block(block)

    encrypted_data = salt + nonce + ciphertext + tag
    blockchain_data = BLOCKCHAIN_HEADER + blockchain.to_json().encode('utf-8')

    out_path = file_path + ".vdec"
    with lzma.open(out_path, 'wb') as f:
        f.write(encrypted_data)
        f.write(blockchain_data)
    if deletemode["mode"] == True:
        delete_pre_file(file_path)
    print(f"✅ Encryption completed: {out_path}")

def cli_decrypt(file_path, password, memo):
    if file_path.endswith(".wav"):
        with open(file_path, 'rb') as f:
            wav_bytes = f.read()
        data = wavencode.wav_bytes_to_binary(wav_bytes)
    else:
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
        print("❌ Error: Decryption failed. The password may be incorrect or the file may be tampered with.")
        sys.exit(1)

    output_file = file_path.replace(".vdec.wav", "_decrypted").replace(".vdec", "_decrypted")
    with open(output_file, 'wb') as f:
        f.write(plaintext)

    username = getpass.getuser()
    file_hash = hashlib.sha256(ciphertext).hexdigest()
    block = Block(file_hash, blockchain.chain[-1].hash if blockchain.chain else "0", "Decrypt", file_hash, username, memo)
    blockchain.add_block(block)

    if not file_path.endswith(".wav"):
        with lzma.open(file_path, 'wb') as f:
            f.write(salt + nonce + ciphertext + tag)
            f.write(BLOCKCHAIN_HEADER)
            f.write(blockchain.to_json().encode('utf-8'))

    print(f"✅ Decryption completed: {output_file}")

def cli_verify_chain(file_path):
    if file_path.endswith(".wav"):
        with open(file_path, 'rb') as f:
            data = wavencode.wav_bytes_to_binary(f.read())
    else:
        with lzma.open(file_path, 'rb') as f:
            data = f.read()
    split_index = data.index(BLOCKCHAIN_HEADER)
    chain_json = data[split_index + len(BLOCKCHAIN_HEADER):].decode('utf-8')
    blockchain = Blockchain.from_json(chain_json)
    if blockchain.is_chain_valid():
        print("✅ Blockchain is consistent")
    else:
        print("❌ Blockchain has inconsistencies")

# --- RSA key check at startup ---
rsa_encryptor.ensure_rsa_keys()

def main():
    parser = argparse.ArgumentParser(description="EncryptSecureDEC CLI")
    parser.add_argument("mode", choices=["encrypt", "decrypt", "verify-chain", "sign", "verify-sign"])
    parser.add_argument("file", help="Target file path")
    parser.add_argument("--memo", default="", help="Operation memo")
    parser.add_argument("--password", help="Password for encryption/decryption (prompted if omitted)")
    parser.add_argument("--delete", action="store_true",help="Delete the plaintext file after encrypting the file.")
    parser.add_argument("--rsa", action="store_true",help="Encrypt / Decrypt RSA Mode")
    parser.add_argument("--pubkey", help="Path to public key file (only required in RSA encrypt mode)")
    args = parser.parse_args()

    # --- validate RSA/pubkey usage ---
        # --- validate RSA/pubkey usage ---
    if args.rsa:
        if args.mode == "decrypt" and args.pubkey:
            parser.error("--pubkey must not be specified in decrypt mode (private key is used automatically)")

    if not args.rsa:
        password = args.password or getpass.getpass("🔑 Enter password: ")

    # Check if file exists
    if not os.path.isfile(args.file):
        print(f"❌ Error: File not found - {args.file}")
        sys.exit(1)

    # For decrypt mode, check extension
    if args.mode == "decrypt" and not (args.file.endswith(".vdec") or args.file.endswith(".rdec")):
        print(f"❌ Error: The file for decryption must have a '.vdec' or '.rdec' extension.")
        sys.exit(1)

    if args.delete:
        deletemode["mode"] = True

    # --- RSA Mode ---
    if args.rsa and args.mode == "encrypt":
        pubkey_path = args.pubkey if args.pubkey else str(rsa_encryptor.RSA_PUB_PATH)
        rsa_encryptor.encrypt_file_with_dialog(args.file, pubkey_path)
        sys.exit()
    elif args.rsa and args.mode == "decrypt":
        rsa_encryptor.decrypt_file_with_dialog(args.file)
        sys.exit()

    # --- Password Mode ---
    if args.mode == "encrypt":
        cli_encrypt(args.file, password, args.memo)
    elif args.mode == "decrypt":
        cli_decrypt(args.file, password, args.memo)
    elif args.mode == "verify-chain":
        cli_verify_chain(args.file)
    elif args.mode == "sign":
        rsa_signer.sign_file(args.file)
    elif args.mode == "verify-sign":
        rsa_signer.verify_file_signature(args.file)
    else:
        print("❌ Unknown mode")

if __name__ == "__main__":
    main()
