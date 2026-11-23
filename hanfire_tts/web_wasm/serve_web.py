#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火三代古音 TTS 浏览器推理与听审反馈本地全栈服务 (serve_web.py)
- 自动注入 COOP / COEP 跨域隔离响应头，激活浏览器 SharedArrayBuffer 多核 WASM SIMD；
- 内置与 Cloudflare Workers + D1 100% 同款的 SQLite REST API：
    * POST /api/feedback              : 公开提交（强制 is_public=0，保存前台用户调音记录）
    * GET  /api/feedback/public       : 公开精选评论墙（仅 is_public=1，剥离 email/admin_note）
    * GET  /api/admin/feedback        : 站长私密查询与统计（Bearer ADMIN_TOKEN）
    * PATCH / DELETE /api/admin/feedback/:id : 审核公开、标状态、写备注、删除
    * GET  /api/admin/export          : 导出 JSON / CSV
"""
from __future__ import annotations

import argparse
import csv
import functools
import hmac
import io
import json
import os
import re
import secrets
import socket
import sqlite3
import ssl
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VALID_ERAS = {"oc", "mc", "em", "em_zyyy"}
VALID_STATUS = {"pending", "adopted", "ignored"}


class FeedbackStore:
    def __init__(self, db_path: Path, schema_path: Path) -> None:
        self.db_path = db_path
        self.lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            if schema_path.exists():
                conn.executescript(schema_path.read_text(encoding="utf-8"))
            else:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                        nickname TEXT NOT NULL,
                        email TEXT NOT NULL DEFAULT '',
                        rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                        ipa_correction TEXT NOT NULL DEFAULT '',
                        comment TEXT NOT NULL DEFAULT '',
                        era TEXT NOT NULL,
                        source_type TEXT NOT NULL DEFAULT 'synth',
                        rhyme_id TEXT NOT NULL DEFAULT '',
                        input_text TEXT NOT NULL,
                        voice_key TEXT NOT NULL DEFAULT 'male_128Hz',
                        speed REAL NOT NULL DEFAULT 1.0,
                        num_step INTEGER NOT NULL DEFAULT 16,
                        compiled_gloss_json TEXT NOT NULL DEFAULT '[]',
                        user_adjustments_json TEXT NOT NULL DEFAULT '[]',
                        model_build_id TEXT NOT NULL DEFAULT '',
                        is_public INTEGER NOT NULL DEFAULT 0 CHECK (is_public IN (0, 1)),
                        status TEXT NOT NULL DEFAULT 'pending',
                        admin_note TEXT NOT NULL DEFAULT ''
                    )"""
                )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _safe_json(raw: str, fallback):
        try:
            return json.loads(raw)
        except Exception:
            return fallback

    @classmethod
    def hydrate(cls, row: sqlite3.Row, include_private: bool = False) -> dict:
        d = dict(row)
        out = {
            "id": d["id"],
            "created_at": d["created_at"],
            "nickname": d["nickname"],
            "rating": d["rating"],
            "ipa_correction": d.get("ipa_correction") or "",
            "comment": d.get("comment") or "",
            "era": d["era"],
            "source_type": d.get("source_type") or "synth",
            "rhyme_id": d.get("rhyme_id") or "",
            "input_text": d.get("input_text") or "",
            "voice_key": d.get("voice_key") or "male_128Hz",
            "speed": d.get("speed", 1.0),
            "num_step": d.get("num_step", 16),
            "user_adjustments": cls._safe_json(d.get("user_adjustments_json") or "[]", []),
            "model_build_id": d.get("model_build_id") or "",
        }
        if include_private:
            out["email"] = d.get("email") or ""
            out["compiled_gloss"] = cls._safe_json(d.get("compiled_gloss_json") or "[]", [])
            out["is_public"] = int(d.get("is_public") or 0)
            out["status"] = d.get("status") or "pending"
            out["admin_note"] = d.get("admin_note") or ""
        return out

    def insert(self, payload: dict) -> int:
        with self.lock, self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO feedback (
                    nickname, email, rating, ipa_correction, comment, era, source_type, rhyme_id,
                    input_text, voice_key, speed, num_step, compiled_gloss_json, user_adjustments_json,
                    model_build_id, is_public, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'pending')""",
                (
                    payload["nickname"],
                    payload["email"],
                    payload["rating"],
                    payload["ipa_correction"],
                    payload["comment"],
                    payload["era"],
                    payload["source_type"],
                    payload["rhyme_id"],
                    payload["input_text"],
                    payload["voice_key"],
                    payload["speed"],
                    payload["num_step"],
                    payload["compiled_gloss_json"],
                    payload["user_adjustments_json"],
                    payload["model_build_id"],
                ),
            )
            return int(cur.lastrowid or 0)

    def list_public(self, era: str = "", limit: int = 30, offset: int = 0) -> dict:
        where = "WHERE is_public = 1 AND era = ?" if era in VALID_ERAS else "WHERE is_public = 1"
        binds = [era] if era in VALID_ERAS else []
        with self.lock, self._conn() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM feedback {where}", binds).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM feedback {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*binds, limit, offset],
            ).fetchall()
        return {"ok": True, "total": int(total), "items": [self.hydrate(r, False) for r in rows]}

    def list_admin(self, params: dict) -> dict:
        era = (params.get("era") or [""])[0]
        rating = (params.get("rating") or [""])[0]
        status = (params.get("status") or [""])[0]
        is_pub = (params.get("is_public") or [""])[0]
        has_fix = (params.get("has_fix") or [""])[0]
        q = (params.get("q") or [""])[0].strip()
        limit = min(200, max(1, int((params.get("limit") or ["100"])[0] or 100)))
        offset = max(0, int((params.get("offset") or ["0"])[0] or 0))

        conds, binds = [], []
        if era in VALID_ERAS:
            conds.append("era = ?")
            binds.append(era)
        if rating == "low":
            conds.append("rating <= 3")
        elif rating in {"1", "2", "3", "4", "5"}:
            conds.append("rating = ?")
            binds.append(int(rating))
        if status in VALID_STATUS:
            conds.append("status = ?")
            binds.append(status)
        if is_pub in {"0", "1"}:
            conds.append("is_public = ?")
            binds.append(int(is_pub))
        if has_fix == "1":
            conds.append("(ipa_correction != '' OR (user_adjustments_json != '' AND user_adjustments_json != '[]'))")
        if q:
            conds.append(
                "(nickname LIKE ? OR email LIKE ? OR input_text LIKE ? OR ipa_correction LIKE ? OR comment LIKE ? OR user_adjustments_json LIKE ?)"
            )
            like = f"%{q}%"
            binds.extend([like] * 6)
        where_sql = f"WHERE {' AND '.join(conds)}" if conds else ""

        with self.lock, self._conn() as conn:
            s = dict(
                conn.execute(
                    """SELECT COUNT(*) AS total,
                              ROUND(AVG(rating), 2) AS avg_rating,
                              SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                              SUM(CASE WHEN status='adopted' THEN 1 ELSE 0 END) AS adopted,
                              SUM(CASE WHEN status='ignored' THEN 1 ELSE 0 END) AS ignored,
                              SUM(CASE WHEN is_public=1 THEN 1 ELSE 0 END) AS public_count,
                              SUM(CASE WHEN ipa_correction != '' OR (user_adjustments_json != '' AND user_adjustments_json != '[]') THEN 1 ELSE 0 END) AS with_fix
                       FROM feedback"""
                ).fetchone()
            )
            by_era = [
                dict(r)
                for r in conn.execute(
                    "SELECT era, COUNT(*) AS cnt, ROUND(AVG(rating), 2) AS avg_rating FROM feedback GROUP BY era"
                ).fetchall()
            ]
            total_f = conn.execute(f"SELECT COUNT(*) FROM feedback {where_sql}", binds).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM feedback {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*binds, limit, offset],
            ).fetchall()

        return {
            "ok": True,
            "stats": {
                "total": int(s["total"] or 0),
                "avg_rating": float(s["avg_rating"] or 0),
                "pending": int(s["pending"] or 0),
                "adopted": int(s["adopted"] or 0),
                "ignored": int(s["ignored"] or 0),
                "public_count": int(s["public_count"] or 0),
                "with_fix": int(s["with_fix"] or 0),
                "by_era": by_era,
            },
            "total_filtered": int(total_f),
            "items": [self.hydrate(r, True) for r in rows],
        }

    def update(self, row_id: int, body: dict):
        sets, binds = [], []
        if "is_public" in body:
            sets.append("is_public = ?")
            binds.append(1 if body["is_public"] else 0)
        if "status" in body:
            if body["status"] not in VALID_STATUS:
                return None, "无效状态"
            sets.append("status = ?")
            binds.append(body["status"])
        if "admin_note" in body:
            sets.append("admin_note = ?")
            binds.append(str(body["admin_note"])[:500])
        if not sets:
            return None, "无可更新字段"
        with self.lock, self._conn() as conn:
            conn.execute(f"UPDATE feedback SET {', '.join(sets)} WHERE id = ?", [*binds, row_id])
            row = conn.execute("SELECT * FROM feedback WHERE id = ?", (row_id,)).fetchone()
        if not row:
            return None, "记录不存在"
        return self.hydrate(row, True), None

    def delete(self, row_id: int) -> None:
        with self.lock, self._conn() as conn:
            conn.execute("DELETE FROM feedback WHERE id = ?", (row_id,))

    def export_all(self) -> list[dict]:
        with self.lock, self._conn() as conn:
            rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()
        return [self.hydrate(r, True) for r in rows]


class CrossOriginIsolatedHandler(SimpleHTTPRequestHandler):
    store: FeedbackStore | None = None
    admin_token: str = ""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".gzbin": "application/octet-stream",
        ".json": "application/json; charset=utf-8",
        ".gz": "application/octet-stream",
        ".wav": "audio/wav",
    }

    def guess_type(self, path: str) -> str:
        if path.endswith(".gzbin") or path.endswith(".gz") or path.endswith(".onnx"):
            return "application/octet-stream"
        if path.endswith(".mjs"):
            return "text/javascript; charset=utf-8"
        if path.endswith(".wasm"):
            return "application/wasm"
        return super().guess_type(path)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Accept-Ranges", "bytes")
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if qs.get("download", [""])[0] == "1":
            fname = Path(parsed.path).name or "hanfire_audio.wav"
            self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
        if "/vendor/ort/" in self.path or any(parsed.path.endswith(ext) for ext in (".gzbin", ".wasm")):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        elif any(parsed.path.endswith(ext) for ext in (".onnx", ".gz")):
            self.send_header("Cache-Control", "public, max-age=3600")
        elif not parsed.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _send_json(self, obj: dict, status: int = 200) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self):
        cl = int(self.headers.get("Content-Length") or 0)
        if cl <= 0 or cl > 256 * 1024:
            return None
        raw = self.rfile.read(cl)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def _check_admin(self) -> bool:
        auth = self.headers.get("Authorization") or ""
        tok = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if not self.admin_token or not tok:
            return False
        return hmac.compare_digest(tok, self.admin_token)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api("GET", parsed)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api("POST", parsed)
            return
        self._send_json({"ok": False, "error": "Method Not Allowed"}, 405)

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api("PATCH", parsed)
            return
        self._send_json({"ok": False, "error": "Method Not Allowed"}, 405)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api("DELETE", parsed)
            return
        self._send_json({"ok": False, "error": "Method Not Allowed"}, 405)

    def _handle_api(self, method: str, parsed) -> None:
        assert self.store is not None
        p = parsed.path
        qs = parse_qs(parsed.query)

        if p == "/api/feedback" and method == "POST":
            body = self._read_json_body()
            if not isinstance(body, dict):
                self._send_json({"ok": False, "error": "JSON 格式错误"}, 400)
                return
            if body.get("hp_website"):
                self._send_json({"ok": True, "id": 0}, 201)
                return
            nickname = str(body.get("nickname") or "").strip()
            email = str(body.get("email") or "").strip()
            try:
                rating = int(body.get("rating"))
            except Exception:
                rating = 0
            ipa_cor = str(body.get("ipa_correction") or "").strip()[:600]
            comment = str(body.get("comment") or "").strip()[:1500]
            era = str(body.get("era") or "mc").strip()
            source_type = "rhyme" if body.get("source_type") == "rhyme" else "synth"
            rhyme_id = str(body.get("rhyme_id") or "").strip()[:80]
            input_text = str(body.get("input_text") or "").strip()[:500]
            voice_key = str(body.get("voice_key") or "male_128Hz").strip()[:40]
            try:
                speed = float(body.get("speed", 1.0))
            except Exception:
                speed = 1.0
            try:
                num_step = int(body.get("num_step", 16))
            except Exception:
                num_step = 16
            build_id = str(body.get("model_build_id") or "").strip()[:40]

            if not nickname or len(nickname) > 40:
                self._send_json({"ok": False, "error": "昵称为必填项（1~40 字）"}, 400)
                return
            if email and (len(email) > 120 or "@" not in email):
                self._send_json({"ok": False, "error": "邮箱格式不正确"}, 400)
                return
            if rating < 1 or rating > 5:
                self._send_json({"ok": False, "error": "请选择 1~5 星评分"}, 400)
                return
            if era not in VALID_ERAS:
                self._send_json({"ok": False, "error": "无效时代参数"}, 400)
                return
            if not input_text:
                self._send_json({"ok": False, "error": "缺少合成文本上下文"}, 400)
                return

            gloss = body.get("compiled_gloss") if isinstance(body.get("compiled_gloss"), list) else []
            adjs = body.get("user_adjustments") if isinstance(body.get("user_adjustments"), list) else []
            rid = self.store.insert(
                {
                    "nickname": nickname,
                    "email": email,
                    "rating": rating,
                    "ipa_correction": ipa_cor,
                    "comment": comment,
                    "era": era,
                    "source_type": source_type,
                    "rhyme_id": rhyme_id,
                    "input_text": input_text,
                    "voice_key": voice_key,
                    "speed": speed,
                    "num_step": num_step,
                    "compiled_gloss_json": json.dumps(gloss[:120], ensure_ascii=False)[:16000],
                    "user_adjustments_json": json.dumps(adjs[:60], ensure_ascii=False)[:12000],
                    "model_build_id": build_id,
                }
            )
            self._send_json({"ok": True, "id": rid}, 201)
            return

        if p == "/api/feedback/public" and method == "GET":
            era = (qs.get("era") or [""])[0]
            limit = min(50, max(1, int((qs.get("limit") or ["30"])[0] or 30)))
            offset = max(0, int((qs.get("offset") or ["0"])[0] or 0))
            self._send_json(self.store.list_public(era=era, limit=limit, offset=offset), 200)
            return

        if p.startswith("/api/admin/"):
            if not self._check_admin():
                self._send_json({"ok": False, "error": "管理员口令无效或未登录"}, 401)
                return

            if p == "/api/admin/feedback" and method == "GET":
                self._send_json(self.store.list_admin(qs), 200)
                return

            if p == "/api/admin/export" and method == "GET":
                fmt = (qs.get("format") or ["json"])[0].lower()
                items = self.store.export_all()
                if fmt == "csv":
                    cols = [
                        "id", "created_at", "nickname", "email", "rating", "era", "source_type", "rhyme_id",
                        "input_text", "ipa_correction", "user_adjustments", "comment", "is_public", "status", "admin_note", "voice_key",
                    ]
                    buf = io.StringIO()
                    w = csv.writer(buf)
                    w.writerow(cols)
                    for it in items:
                        w.writerow([
                            json.dumps(it.get(c, []), ensure_ascii=False) if c == "user_adjustments" else it.get(c, "")
                            for c in cols
                        ])
                    raw = ("\ufeff" + buf.getvalue()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="hanfire_feedback.csv"')
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                raw = json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="hanfire_feedback.json"')
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            m = re.match(r"^/api/admin/feedback/(\d+)$", p)
            if m:
                rid = int(m.group(1))
                if method == "PATCH":
                    body = self._read_json_body()
                    if not isinstance(body, dict):
                        self._send_json({"ok": False, "error": "JSON 格式错误"}, 400)
                        return
                    item, err = self.store.update(rid, body)
                    if err:
                        self._send_json({"ok": False, "error": err}, 404 if err == "记录不存在" else 400)
                        return
                    self._send_json({"ok": True, "item": item}, 200)
                    return
                if method == "DELETE":
                    self.store.delete(rid)
                    self._send_json({"ok": True, "deleted": rid}, 200)
                    return

        self._send_json({"ok": False, "error": "Not Found"}, 404)


def load_or_create_local_admin_token(base_dir: Path) -> str:
    env_tok = os.environ.get("HANFIRE_ADMIN_TOKEN", "").strip()
    if env_tok:
        return env_tok
    tok_file = base_dir / ".local_admin_token"
    if tok_file.exists():
        val = tok_file.read_text(encoding="utf-8").strip()
        if val:
            return val
    val = "hf-" + secrets.token_hex(12)
    tok_file.write_text(val + "\n", encoding="utf-8")
    try:
        os.chmod(tok_file, 0o600)
    except Exception:
        pass
    return val


def ensure_local_tls_cert(base_dir: Path) -> tuple[Path, Path] | None:
    tls_dir = base_dir / ".local_tls"
    cert_p = tls_dir / "local.crt"
    key_p = tls_dir / "local.key"
    if cert_p.exists() and key_p.exists():
        return cert_p, key_p
    tls_dir.mkdir(parents=True, exist_ok=True)
    # 自动收集本机局域网 IP 写入 SAN，避免浏览器因缺 SAN 拒连
    ips = {"127.0.0.1"}
    try:
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True, timeout=3)
        for m in re.finditer(r"inet\s+(\d+\.\d+\.\d+\.\d+)/", out):
            ips.add(m.group(1))
    except Exception:
        try:
            ips.add(socket.gethostbyname(socket.gethostname()))
        except Exception:
            pass
    san = ",".join(["DNS:localhost"] + [f"IP:{ip}" for ip in sorted(ips)])
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key_p), "-out", str(cert_p),
                "-days", "365", "-subj", "/CN=hanfire-local",
                "-addext", f"subjectAltName={san}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return cert_p, key_p
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="启动汉火古音 QAT-INT8 WASM 浏览器推理与听审反馈全栈服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP 监听端口 (默认 8080)")
    parser.add_argument("--https-port", type=int, default=8444, help="HTTPS 监听端口 (默认 8444，设为 0 关闭；局域网开启多核 SharedArrayBuffer 必需)")
    parser.add_argument("--dir", default="", help="静态根目录 (默认优先使用 dist_cf 若存在，否则当前目录)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    if args.dir:
        web_dir = (base_dir / args.dir).resolve() if not Path(args.dir).is_absolute() else Path(args.dir).resolve()
    else:
        cf_dir = base_dir / "dist_cf"
        web_dir = cf_dir if cf_dir.exists() else base_dir

    store = FeedbackStore(base_dir / "feedback.db", base_dir / "schema.sql")
    admin_tok = load_or_create_local_admin_token(base_dir)

    class BoundHandler(CrossOriginIsolatedHandler):
        pass

    BoundHandler.store = store
    BoundHandler.admin_token = admin_tok

    class HighConcurrencyHTTPServer(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True

    class HighConcurrencyHTTPSServer(HighConcurrencyHTTPServer):
        def __init__(self, server_address, RequestHandlerClass, ssl_ctx: ssl.SSLContext):
            self.ssl_ctx = ssl_ctx
            super().__init__(server_address, RequestHandlerClass)

        def process_request_thread(self, request, client_address):
            try:
                tls_sock = self.ssl_ctx.wrap_socket(request, server_side=True)
            except (ssl.SSLError, OSError):
                self.shutdown_request(request)
                return
            try:
                super().process_request_thread(tls_sock, client_address)
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                pass

    handler = functools.partial(BoundHandler, directory=str(web_dir))
    http_server = HighConcurrencyHTTPServer((args.host, args.port), handler)

    https_server = None
    if args.https_port > 0:
        pair = ensure_local_tls_cert(base_dir)
        if pair:
            cert_p, key_p = pair
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=str(cert_p), keyfile=str(key_p))
            https_server = HighConcurrencyHTTPSServer((args.host, args.https_port), handler, ctx)
            threading.Thread(target=https_server.serve_forever, daemon=True).start()

    print(f"🔥 汉火古音 QAT-INT8 WASM 听审工作台已启动:")
    print(f"   👉 HTTP  监听: http://{args.host}:{args.port}/ (本机: http://127.0.0.1:{args.port}/)")
    if https_server:
        print(f"   👉 HTTPS 监听: https://{args.host}:{args.https_port}/ (局域网访问此 HTTPS 地址可立即开启多核 WASM SIMD!)")
    print(f"   👉 站长后台:   http://127.0.0.1:{args.port}/admin.html")
    print(f"   👉 本地管理口令 (已存 .local_admin_token): {admin_tok}")
    print(f"   👉 静态目录:   {web_dir}")
    print(f"   👉 本地反馈库: {base_dir / 'feedback.db'}")
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止服务。")
        if https_server:
            https_server.shutdown()


if __name__ == "__main__":
    main()
