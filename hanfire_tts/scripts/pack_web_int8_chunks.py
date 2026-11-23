#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音 TTS —— Web INT8 分片压缩打包与 Cloudflare Workers 发布目录构建工具
(pack_web_int8_chunks.py)

功能：
1. 将三大 QAT-INT8 ONNX 模型 (text_encoder_int8 / fm_decoder_int8 / vocos_spec_int8)
   按 16 MiB 切分原始字节并逐块 deterministic gzip -6 压缩为 .gzbin 分片，输出至 web_wasm/models_chunks/；
   - 单个分片严格 < 15 MiB（远低于 Cloudflare Workers Static Assets 25 MiB 单文件上限，零 R2 即可直接部署！）；
   - 使用 .gzbin 后缀避免 CDN/代理自动附加 Content-Encoding 触发二次解压冲突；
   - 生成 models_chunks/manifest.json（含每模型版本号、每分片 offset / rawSize / gzSize / rawSha256）；
   - 自动执行全量解压拼接 + SHA-256 逐字节回环自检，确保 100% 无损；
2. 抽取最小 ONNX Runtime WASM 运行时 3 文件 (ort.wasm.bundle.min.mjs + ort-wasm-simd-threaded.{mjs,wasm})
   至 web_wasm/vendor/ort/（脱离 node_modules 依赖）；
3. 构建纯净 Cloudflare Workers INT8 生产发布目录 web_wasm/dist_cf/（仅含 INT8 分片与前端必需资源，不含 525MB FP32 模型）。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE.parent / "web_wasm"
MODELS_DIR = WEB_DIR / "models"
CHUNKS_DIR = WEB_DIR / "models_chunks"
VENDOR_ORT_DIR = WEB_DIR / "vendor" / "ort"
DIST_CF_DIR = WEB_DIR / "dist_cf"

CHUNK_RAW_BYTES = 16 * 1024 * 1024  # 16 MiB 原始分片 -> gzip 后约 13.5~14.5 MiB (< 25 MiB CF 限制)
MAX_CF_ASSET_BYTES = 24 * 1024 * 1024

INT8_MODELS = [
    ("te", "text_encoder_int8.onnx"),
    ("fm", "fm_decoder_int8.onnx"),
    ("voc", "vocos_spec_int8.onnx"),
]

ORT_PLAIN_FILES = [
    "ort.webgpu.bundle.min.mjs",
    "ort-wasm-simd-threaded.asyncify.mjs",
]
ORT_GZIP_WASM_FILE = "ort-wasm-simd-threaded.asyncify.wasm"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync_minimal_ort_runtime() -> None:
    src_dist = WEB_DIR / "node_modules" / "onnxruntime-web" / "dist"
    if VENDOR_ORT_DIR.exists():
        shutil.rmtree(VENDOR_ORT_DIR)
    VENDOR_ORT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in ORT_PLAIN_FILES:
        src = src_dist / fn
        if not src.exists():
            raise FileNotFoundError(f"缺少 ONNX Runtime WebGPU/WASM 运行时文件: {src}")
        dst = VENDOR_ORT_DIR / fn
        shutil.copy2(src, dst)
        sz = dst.stat().st_size
        assert sz < MAX_CF_ASSET_BYTES, f"文件超过 Cloudflare 25 MiB 限制: {fn} ({sz / 1e6:.2f} MB)"
    # 将 28.7 MB 的 asyncify.wasm 压缩为单个 ~7.2 MB 的 .gzbin（远低于 Cloudflare 25 MiB 上限，前端原生 gzip 秒解压注入 wasmBinary）
    wasm_src = src_dist / ORT_GZIP_WASM_FILE
    if not wasm_src.exists():
        raise FileNotFoundError(f"缺少 ONNX Runtime WASM 二进制: {wasm_src}")
    wasm_raw = wasm_src.read_bytes()
    wasm_gz = gzip.compress(wasm_raw, compresslevel=6, mtime=0)
    assert len(wasm_gz) < MAX_CF_ASSET_BYTES, f"压缩后 WASM 超过 25 MiB: {len(wasm_gz)/1e6:.2f} MB"
    assert gzip.decompress(wasm_gz) == wasm_raw
    (VENDOR_ORT_DIR / f"{ORT_GZIP_WASM_FILE}.gzbin").write_bytes(wasm_gz)
    print(
        f"  ✓ 最小 ORT WebGPU+WASM 运行时已同步至 {VENDOR_ORT_DIR.relative_to(WEB_DIR)} "
        f"({len(ORT_PLAIN_FILES)} JS + {ORT_GZIP_WASM_FILE}.gzbin {len(wasm_raw)/1e6:.1f}MB->{len(wasm_gz)/1e6:.2f}MB)"
    )


def pack_int8_chunks() -> dict:
    if CHUNKS_DIR.exists():
        shutil.rmtree(CHUNKS_DIR)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_models = {}
    total_raw = 0
    total_gz = 0
    combined_hash = hashlib.sha256()

    for key, fname in INT8_MODELS:
        src = MODELS_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"缺少 INT8 模型文件: {src}，请先运行 export_web_onnx.py")
        raw = src.read_bytes()
        full_sha = sha256_hex(raw)
        combined_hash.update(full_sha.encode("ascii"))
        version = full_sha[:16]

        chunks_meta = []
        model_gz_total = 0
        rebuilt = bytearray(len(raw))
        offset = 0
        idx = 0
        while offset < len(raw):
            part_raw = raw[offset : offset + CHUNK_RAW_BYTES]
            part_sha = sha256_hex(part_raw)
            part_gz = gzip.compress(part_raw, compresslevel=6, mtime=0)
            assert len(part_gz) < MAX_CF_ASSET_BYTES, (
                f"分片超过 Cloudflare 25 MiB 限制: {key}.part{idx:02d} ({len(part_gz)/1e6:.2f} MB)"
            )
            out_name = f"{key}.{version}.part{idx:02d}.gzbin"
            (CHUNKS_DIR / out_name).write_bytes(part_gz)

            # 无损回环自检
            dec = gzip.decompress(part_gz)
            assert dec == part_raw and sha256_hex(dec) == part_sha
            rebuilt[offset : offset + len(part_raw)] = dec

            chunks_meta.append({
                "index": idx,
                "file": out_name,
                "offset": offset,
                "rawSize": len(part_raw),
                "gzSize": len(part_gz),
                "rawSha256": part_sha,
            })
            model_gz_total += len(part_gz)
            offset += len(part_raw)
            idx += 1

        assert bytes(rebuilt) == raw and sha256_hex(bytes(rebuilt)) == full_sha
        manifest_models[key] = {
            "name": fname,
            "version": version,
            "sha256": full_sha,
            "totalSize": len(raw),
            "totalGzSize": model_gz_total,
            "chunks": chunks_meta,
        }
        total_raw += len(raw)
        total_gz += model_gz_total
        print(
            f"  ✓ [{key:3s}] {fname:24s} | raw={len(raw)/1e6:6.2f} MB -> gz={model_gz_total/1e6:6.2f} MB "
            f"({100.0*model_gz_total/len(raw):4.1f}%) | {len(chunks_meta)} 分片 | ver={version}"
        )

    manifest = {
        "format": "hanfire-chunked-gzip-v1",
        "quantization": "weight-only-int8-nbits",
        "chunkRawBytes": CHUNK_RAW_BYTES,
        "buildId": combined_hash.hexdigest()[:16],
        "totalRawBytes": total_raw,
        "totalGzBytes": total_gz,
        "models": manifest_models,
    }
    (CHUNKS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"  ✓ 分片清单已生成: models_chunks/manifest.json | "
        f"总计 {total_raw/1e6:.2f} MB -> {total_gz/1e6:.2f} MB ({sum(len(m['chunks']) for m in manifest_models.values())} 分片, 逐字节回环自检 PASS ✅)"
    )
    return manifest


def build_cloudflare_dist() -> None:
    if DIST_CF_DIR.exists():
        shutil.rmtree(DIST_CF_DIR)
    DIST_CF_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 核心前端文件（工作台 + 站长私密后台 + 站点图标 + DSP/Worker/分片缓存）
    for fn in (
        "index.html", "admin.html", "favicon.ico", "favicon.svg", "apple-touch-icon.png",
        "hanfire_dsp.mjs", "hanfire_worker.mjs", "model_cache.mjs",
    ):
        src = WEB_DIR / fn
        if src.exists():
            shutil.copy2(src, DIST_CF_DIR / fn)

    # 2) 最小 ORT WASM 运行时
    shutil.copytree(VENDOR_ORT_DIR, DIST_CF_DIR / "vendor" / "ort")

    # 3) 前端静态资产（压缩词表 .gz + 24kHz 参考音色锚 + 三代全韵系试听索引）
    dst_assets = DIST_CF_DIR / "assets"
    dst_assets.mkdir(parents=True, exist_ok=True)
    for fn in ("lexicon_3era.json.gz", "voice_prompts.json", "rhyme_catalog.json"):
        src = WEB_DIR / "assets" / fn
        if not src.exists():
            raise FileNotFoundError(f"缺少前端资产: {src}")
        shutil.copy2(src, dst_assets / fn)

    # 4) 分片压缩 INT8 模型
    shutil.copytree(CHUNKS_DIR, DIST_CF_DIR / "models_chunks")

    # 5) 校验 dist_cf 中所有文件均满足 Cloudflare Workers Static Assets < 25 MiB 限制
    max_sz = 0
    max_name = ""
    total_sz = 0
    n_files = 0
    for p in DIST_CF_DIR.rglob("*"):
        if p.is_file():
            sz = p.stat().st_size
            total_sz += sz
            n_files += 1
            if sz > max_sz:
                max_sz = sz
                max_name = str(p.relative_to(DIST_CF_DIR))
            assert sz < MAX_CF_ASSET_BYTES, f"超出 Cloudflare 单文件 25 MiB 限制: {p} ({sz/1e6:.2f} MB)"

    print(
        f"  ✓ Cloudflare Workers INT8 生产包构建完成: web_wasm/dist_cf/ "
        f"({n_files} 文件, 总计 {total_sz/1e6:.2f} MB, 最大单文件 {max_name}={max_sz/1e6:.2f} MB < 25 MiB ✅)"
    )


def main() -> None:
    print("=== 打包汉火古音 Web QAT-INT8 分片压缩与 Cloudflare Workers 发布包 ===")
    sync_minimal_ort_runtime()
    pack_int8_chunks()
    build_cloudflare_dist()
    print("✅ 全部打包完成！")


if __name__ == "__main__":
    main()
