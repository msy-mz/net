# Filename: base.py
# Path: inf/payload/extract/base.py
# Description: 提取流的通用保存工具，包含 save_bin_payloads 与 meta.json 写入
# Author: msy
# Date: 2025

import os
import json

# 通用保存函数：保存 bin 文件 + meta.json
def save_bin_payloads(flows, output_dir, is_valid_fn=None, max_payload_len=1024):
    os.makedirs(output_dir, exist_ok=True)
    bin_files = []
    meta = {}
    filtered, total_payload = 0, 0

    for idx, (fid, pkt_list) in enumerate(flows.items()):
        pkt_list.sort(key=lambda x: x[0])
        payload = b"".join([p for _, p in pkt_list])
        if len(payload) == 0:
            continue

        if is_valid_fn and not is_valid_fn(payload):
            filtered += 1
            continue

        payload = payload[:max_payload_len]
        fname = f"flow_{idx}.bin"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "wb") as f:
            f.write(payload)

        ts = pkt_list[0][1] if len(pkt_list[0]) > 1 else None
        meta[fname] = {
            "fid": fid,
            "timestamp": ts,
            "payload_len": len(payload)
        }
        bin_files.append(fpath)
        total_payload += len(payload)

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    stats = {
        "saved_flows": len(bin_files),
        "filtered_flows": filtered,
        "total_payload_bytes": total_payload
    }
    return bin_files, stats
