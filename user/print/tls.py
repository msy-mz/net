# Filename: tls.py
# Path: user/print/tls.py
# Description: 提供 TLS Client Hello 消息的字段提取与 JA3 指纹生成能力
# Author: msy
# Date: 2025

import hashlib

# 对字符串进行 MD5 哈希
def hash_str(s):
    return hashlib.md5(s.encode()).hexdigest() if s else ""

# 解析 TLS Client Hello 消息，提取 Cipher Suites
def parse_tls_client_hello(payload):
    try:
        if len(payload) < 5 or payload[0] != 0x16 or payload[1] != 0x03:
            return None

        version = int.from_bytes(payload[1:3], byteorder="big")
        ciphers_len = int.from_bytes(payload[43:45], byteorder="big")
        ciphers = payload[45:45 + ciphers_len]

        extensions_len = int.from_bytes(payload[45 + ciphers_len:47 + ciphers_len], byteorder="big")
        extensions = payload[47 + ciphers_len:47 + ciphers_len + extensions_len]

        return ciphers
    except Exception:
        return None

# 解析 TLS Client Hello 消息并生成 JA3 指纹
def parse_ja3(payload):
    try:
        if len(payload) < 5 or payload[0] != 0x16 or payload[1] != 0x03:
            return None

        version = int.from_bytes(payload[1:3], byteorder="big")
        ciphers_len = int.from_bytes(payload[43:45], byteorder="big")
        ciphers = payload[45:45 + ciphers_len]

        extensions_len = int.from_bytes(payload[45 + ciphers_len:47 + ciphers_len], byteorder="big")
        extensions = payload[47 + ciphers_len:47 + ciphers_len + extensions_len]

        ja3_str = f"{version}-{len(ciphers) // 2}-"
        ja3_str += "-".join([f"{ciphers[i]:02x}{ciphers[i+1]:02x}" for i in range(0, len(ciphers), 2)])
        ja3_str += "-" + str(len(extensions))
        ja3_str += "-" + "-".join([str(ext) for ext in extensions])

        return hash_str(ja3_str)
    except Exception:
        return None
