/**
 * 汉火古音 TTS —— Cloudflare Workers 全栈入口 (cf_worker.mjs)
 *
 * 1. 静态资源 + 自动 COOP/COEP/CORP 响应头 + .gzbin/.wasm 永久强缓存（可选 R2 绑定）；
 * 2. 听审打分与众包纠错 REST API（基于 Cloudflare D1 SQLite）：
 *    - POST /api/feedback          : 公开提交（强制 is_public=0，保存前台用户调音轨迹 user_adjustments）
 *    - GET  /api/feedback/public   : 公开精选评论墙（仅返回站长审核 is_public=1 的条目，严格剥离 email/admin_note）
 *    - GET  /api/admin/feedback    : 站长私密查询与统计（Bearer ADMIN_TOKEN 恒定时间校验）
 *    - PATCH/DELETE /api/admin/feedback/:id : 公开/取消公开、标状态、备注、删除
 *    - GET  /api/admin/export      : 站长导出 JSON / CSV
 */

const VALID_ERAS = new Set(["oc", "mc", "em", "em_zyyy"]);
const VALID_STATUS = new Set(["pending", "adopted", "ignored"]);

function jsonResp(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...extraHeaders,
    },
  });
}

function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  const enc = new TextEncoder();
  const ba = enc.encode(a);
  const bb = enc.encode(b);
  if (ba.length !== bb.length) return false;
  let diff = 0;
  for (let i = 0; i < ba.length; i++) diff |= ba[i] ^ bb[i];
  return diff === 0;
}

function verifyAdmin(request, env) {
  const expected = env.ADMIN_TOKEN;
  if (!expected) return { ok: false, status: 503, msg: "尚未在 Cloudflare Workers 配置 ADMIN_TOKEN Secret" };
  const auth = request.headers.get("Authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
  if (!token || !timingSafeEqual(token, expected)) {
    return { ok: false, status: 401, msg: "管理员口令无效或已失效" };
  }
  return { ok: true };
}

function safeJsonParse(str, fallback) {
  try {
    return JSON.parse(str);
  } catch (_) {
    return fallback;
  }
}

function hydrateRow(row, includePrivate = false) {
  const base = {
    id: row.id,
    created_at: row.created_at,
    nickname: row.nickname,
    rating: row.rating,
    ipa_correction: row.ipa_correction || "",
    comment: row.comment || "",
    era: row.era,
    source_type: row.source_type || "synth",
    rhyme_id: row.rhyme_id || "",
    input_text: row.input_text || "",
    voice_key: row.voice_key || "male_128Hz",
    speed: row.speed ?? 1.0,
    num_step: row.num_step ?? 16,
    user_adjustments: safeJsonParse(row.user_adjustments_json || "[]", []),
    model_build_id: row.model_build_id || "",
  };
  if (includePrivate) {
    base.email = row.email || "";
    base.compiled_gloss = safeJsonParse(row.compiled_gloss_json || "[]", []);
    base.is_public = Number(row.is_public || 0);
    base.status = row.status || "pending";
    base.admin_note = row.admin_note || "";
  }
  return base;
}

function toCsvCell(v) {
  const s = v == null ? "" : String(v);
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

async function ensureSchema(db) {
  await db
    .prepare(
      `CREATE TABLE IF NOT EXISTS feedback (
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
      )`
    )
    .run();
}

async function handleApi(request, env, url) {
  const db = env.FEEDBACK_DB;
  if (!db) {
    return jsonResp(
      { ok: false, error: "尚未在 wrangler.toml 绑定 Cloudflare D1 数据库 (FEEDBACK_DB)" },
      503
    );
  }
  await ensureSchema(db);
  const path = url.pathname;
  const method = request.method.toUpperCase();

  // 1) 公开提交反馈（强制 is_public=0）
  if (path === "/api/feedback" && method === "POST") {
    let body;
    try {
      body = await request.json();
    } catch (_) {
      return jsonResp({ ok: false, error: "JSON 格式错误" }, 400);
    }
    if (body.hp_website) {
      return jsonResp({ ok: true, id: 0 }, 201);
    }
    const nickname = String(body.nickname || "").trim();
    const email = String(body.email || "").trim();
    const rating = parseInt(body.rating, 10);
    const ipaCorrection = String(body.ipa_correction || "").trim().slice(0, 600);
    const comment = String(body.comment || "").trim().slice(0, 1500);
    const era = String(body.era || "mc").trim();
    const sourceType = body.source_type === "rhyme" ? "rhyme" : "synth";
    const rhymeId = String(body.rhyme_id || "").trim().slice(0, 80);
    const inputText = String(body.input_text || "").trim().slice(0, 500);
    const voiceKey = String(body.voice_key || "male_128Hz").trim().slice(0, 40);
    const speed = Number.isFinite(Number(body.speed)) ? Number(body.speed) : 1.0;
    const numStep = parseInt(body.num_step, 10) || 16;
    const modelBuildId = String(body.model_build_id || "").trim().slice(0, 40);

    if (!nickname || nickname.length > 40) {
      return jsonResp({ ok: false, error: "昵称为必填项（1~40 字）" }, 400);
    }
    if (email && (email.length > 120 || !email.includes("@"))) {
      return jsonResp({ ok: false, error: "邮箱格式不正确" }, 400);
    }
    if (!Number.isInteger(rating) || rating < 1 || rating > 5) {
      return jsonResp({ ok: false, error: "请选择 1~5 星评分" }, 400);
    }
    if (!VALID_ERAS.has(era)) {
      return jsonResp({ ok: false, error: "无效时代参数" }, 400);
    }
    if (!inputText) {
      return jsonResp({ ok: false, error: "缺少合成文本上下文" }, 400);
    }

    const glossArr = Array.isArray(body.compiled_gloss) ? body.compiled_gloss.slice(0, 120) : [];
    const adjArr = Array.isArray(body.user_adjustments) ? body.user_adjustments.slice(0, 60) : [];
    const glossJson = JSON.stringify(glossArr).slice(0, 16000);
    const adjJson = JSON.stringify(adjArr).slice(0, 12000);

    const res = await db
      .prepare(
        `INSERT INTO feedback (
          nickname, email, rating, ipa_correction, comment, era, source_type, rhyme_id,
          input_text, voice_key, speed, num_step, compiled_gloss_json, user_adjustments_json,
          model_build_id, is_public, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'pending')`
      )
      .bind(
        nickname, email, rating, ipaCorrection, comment, era, sourceType, rhymeId,
        inputText, voiceKey, speed, numStep, glossJson, adjJson, modelBuildId
      )
      .run();

    return jsonResp({ ok: true, id: res.meta?.last_row_id || 0 }, 201);
  }

  // 2) 公开精选评论墙（仅 is_public=1，严格剥离 email/admin_note）
  if (path === "/api/feedback/public" && method === "GET") {
    const era = url.searchParams.get("era") || "";
    const limit = Math.min(50, Math.max(1, parseInt(url.searchParams.get("limit") || "30", 10)));
    const offset = Math.max(0, parseInt(url.searchParams.get("offset") || "0", 10));

    const where = era && VALID_ERAS.has(era) ? "WHERE is_public = 1 AND era = ?" : "WHERE is_public = 1";
    const binds = era && VALID_ERAS.has(era) ? [era] : [];

    const cntRow = await db.prepare(`SELECT COUNT(*) AS cnt FROM feedback ${where}`).bind(...binds).first();
    const rows = await db
      .prepare(`SELECT * FROM feedback ${where} ORDER BY id DESC LIMIT ? OFFSET ?`)
      .bind(...binds, limit, offset)
      .all();

    return jsonResp({
      ok: true,
      total: Number(cntRow?.cnt || 0),
      items: (rows.results || []).map((r) => hydrateRow(r, false)),
    });
  }

  // 3) 以下全部为站长私密接口，强制鉴权
  if (path.startsWith("/api/admin/")) {
    const auth = verifyAdmin(request, env);
    if (!auth.ok) return jsonResp({ ok: false, error: auth.msg }, auth.status);

    if (path === "/api/admin/feedback" && method === "GET") {
      const era = url.searchParams.get("era") || "";
      const rating = url.searchParams.get("rating") || "";
      const status = url.searchParams.get("status") || "";
      const isPublic = url.searchParams.get("is_public") || "";
      const hasFix = url.searchParams.get("has_fix") || "";
      const q = (url.searchParams.get("q") || "").trim();
      const limit = Math.min(200, Math.max(1, parseInt(url.searchParams.get("limit") || "100", 10)));
      const offset = Math.max(0, parseInt(url.searchParams.get("offset") || "0", 10));

      const conds = [];
      const binds = [];
      if (era && VALID_ERAS.has(era)) { conds.push("era = ?"); binds.push(era); }
      if (rating === "low") { conds.push("rating <= 3"); }
      else if (/^[1-5]$/.test(rating)) { conds.push("rating = ?"); binds.push(parseInt(rating, 10)); }
      if (status && VALID_STATUS.has(status)) { conds.push("status = ?"); binds.push(status); }
      if (isPublic === "0" || isPublic === "1") { conds.push("is_public = ?"); binds.push(parseInt(isPublic, 10)); }
      if (hasFix === "1") {
        conds.push("(ipa_correction != '' OR (user_adjustments_json != '' AND user_adjustments_json != '[]'))");
      }
      if (q) {
        conds.push("(nickname LIKE ? OR email LIKE ? OR input_text LIKE ? OR ipa_correction LIKE ? OR comment LIKE ? OR user_adjustments_json LIKE ?)");
        const like = `%${q}%`;
        binds.push(like, like, like, like, like, like);
      }
      const whereSql = conds.length ? `WHERE ${conds.join(" AND ")}` : "";

      const summary = await db
        .prepare(
          `SELECT COUNT(*) AS total,
                  ROUND(AVG(rating), 2) AS avg_rating,
                  SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                  SUM(CASE WHEN status='adopted' THEN 1 ELSE 0 END) AS adopted,
                  SUM(CASE WHEN status='ignored' THEN 1 ELSE 0 END) AS ignored,
                  SUM(CASE WHEN is_public=1 THEN 1 ELSE 0 END) AS public_count,
                  SUM(CASE WHEN ipa_correction != '' OR (user_adjustments_json != '' AND user_adjustments_json != '[]') THEN 1 ELSE 0 END) AS with_fix
           FROM feedback`
        )
        .first();
      const byEraRes = await db
        .prepare("SELECT era, COUNT(*) AS cnt, ROUND(AVG(rating), 2) AS avg_rating FROM feedback GROUP BY era")
        .all();
      const filtCnt = await db.prepare(`SELECT COUNT(*) AS cnt FROM feedback ${whereSql}`).bind(...binds).first();
      const rows = await db
        .prepare(`SELECT * FROM feedback ${whereSql} ORDER BY id DESC LIMIT ? OFFSET ?`)
        .bind(...binds, limit, offset)
        .all();

      return jsonResp({
        ok: true,
        stats: {
          total: Number(summary?.total || 0),
          avg_rating: Number(summary?.avg_rating || 0),
          pending: Number(summary?.pending || 0),
          adopted: Number(summary?.adopted || 0),
          ignored: Number(summary?.ignored || 0),
          public_count: Number(summary?.public_count || 0),
          with_fix: Number(summary?.with_fix || 0),
          by_era: byEraRes.results || [],
        },
        total_filtered: Number(filtCnt?.cnt || 0),
        items: (rows.results || []).map((r) => hydrateRow(r, true)),
      });
    }

    if (path === "/api/admin/export" && method === "GET") {
      const fmt = (url.searchParams.get("format") || "json").toLowerCase();
      const rows = await db.prepare("SELECT * FROM feedback ORDER BY id DESC").all();
      const items = (rows.results || []).map((r) => hydrateRow(r, true));
      if (fmt === "csv") {
        const cols = [
          "id", "created_at", "nickname", "email", "rating", "era", "source_type", "rhyme_id",
          "input_text", "ipa_correction", "user_adjustments", "comment", "is_public", "status", "admin_note", "voice_key",
        ];
        const lines = [cols.join(",")];
        for (const it of items) {
          lines.push(
            cols
              .map((c) => (c === "user_adjustments" ? toCsvCell(JSON.stringify(it[c] || [])) : toCsvCell(it[c])))
              .join(",")
          );
        }
        return new Response("\uFEFF" + lines.join("\r\n"), {
          status: 200,
          headers: {
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": 'attachment; filename="hanfire_feedback.csv"',
            "Cache-Control": "no-store",
          },
        });
      }
      return new Response(JSON.stringify(items, null, 2), {
        status: 200,
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          "Content-Disposition": 'attachment; filename="hanfire_feedback.json"',
          "Cache-Control": "no-store",
        },
      });
    }

    const mId = path.match(/^\/api\/admin\/feedback\/(\d+)$/);
    if (mId) {
      const id = parseInt(mId[1], 10);
      if (method === "PATCH") {
        let body;
        try {
          body = await request.json();
        } catch (_) {
          return jsonResp({ ok: false, error: "JSON 格式错误" }, 400);
        }
        const sets = [];
        const binds = [];
        if (body.is_public !== undefined) {
          sets.push("is_public = ?");
          binds.push(body.is_public ? 1 : 0);
        }
        if (body.status !== undefined) {
          if (!VALID_STATUS.has(body.status)) return jsonResp({ ok: false, error: "无效状态" }, 400);
          sets.push("status = ?");
          binds.push(body.status);
        }
        if (body.admin_note !== undefined) {
          sets.push("admin_note = ?");
          binds.push(String(body.admin_note).slice(0, 500));
        }
        if (sets.length === 0) return jsonResp({ ok: false, error: "无可更新字段" }, 400);
        await db.prepare(`UPDATE feedback SET ${sets.join(", ")} WHERE id = ?`).bind(...binds, id).run();
        const updated = await db.prepare("SELECT * FROM feedback WHERE id = ?").bind(id).first();
        if (!updated) return jsonResp({ ok: false, error: "记录不存在" }, 404);
        return jsonResp({ ok: true, item: hydrateRow(updated, true) });
      }
      if (method === "DELETE") {
        await db.prepare("DELETE FROM feedback WHERE id = ?").bind(id).run();
        return jsonResp({ ok: true, deleted: id });
      }
    }
  }

  return jsonResp({ ok: false, error: "Not Found" }, 404);
}

// 进程内 ETag 缓存：对未自带 ETag 的入口 HTML/清单按内容 SHA-256 计算弱 ETag（每个 Worker 隔离区仅算一次）
const _computedEtagCache = new Map();
// 每次发版更新此时间戳（HTTP-date 格式），配合 Last-Modified + If-Modified-Since，
// 即使 Cloudflare 边缘对 text/html 做动态压缩/邮箱混淆剥离了 ETag，仍能 100% 命中 304 Not Modified！
const BUILD_LAST_MODIFIED = "Thu, 19 Nov 2026 14:20:00 GMT";
const BUILD_LAST_MODIFIED_MS = Date.parse(BUILD_LAST_MODIFIED);

async function computeWeakEtagFromArrayBuffer(buf) {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const arr = new Uint8Array(digest);
  let hex = "";
  for (let i = 0; i < 16; i++) hex += arr[i].toString(16).padStart(2, "0");
  // 使用弱 ETag (W/"...")：防止 Cloudflare 边缘动态 Brotli/Gzip 压缩时自动剥离强 ETag
  return `W/"${hex}"`;
}

function ifNoneMatchMatches(inmHeader, etag) {
  if (!inmHeader || !etag) return false;
  const clean = (s) => s.trim().replace(/^W\//i, "");
  const target = clean(etag);
  return inmHeader.split(",").some((part) => {
    const t = part.trim();
    return t === "*" || clean(t) === target;
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith("/api/")) {
      const apiResp = await handleApi(request, env, url);
      if (url.pathname === "/api/feedback/public" && request.method === "GET" && apiResp.ok) {
        const h = new Headers(apiResp.headers);
        h.set("Cache-Control", "public, max-age=15, stale-while-revalidate=60");
        return new Response(apiResp.body, { status: apiResp.status, statusText: apiResp.statusText, headers: h });
      }
      return apiResp;
    }

    let resp = null;
    if (env.MODEL_BUCKET && url.pathname.startsWith("/models_chunks/")) {
      const key = url.pathname.slice(1);
      const obj = await env.MODEL_BUCKET.get(key);
      if (obj) {
        const h = new Headers();
        obj.writeHttpMetadata(h);
        if (obj.httpEtag) h.set("ETag", obj.httpEtag);
        h.set(
          "Content-Type",
          url.pathname.endsWith(".json") ? "application/json; charset=utf-8" : "application/octet-stream"
        );
        resp = new Response(obj.body, { status: 200, headers: h });
      }
    }

    if (!resp) {
      // 剥离自定义 ?v= 版本参数后再取静态资源（防止部分环境误判路径），外层再附加 1 年 immutable 强缓存
      const assetReq = url.searchParams.has("v")
        ? new Request(new URL(url.pathname, url.origin).toString(), {
            method: request.method,
            headers: request.headers,
          })
        : request;
      resp = await env.ASSETS.fetch(assetReq);
    }

    const headers = new Headers(resp.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("Cross-Origin-Resource-Policy", "same-origin");

    const p = url.pathname;
    const isVersioned = url.searchParams.has("v");
    const isHtmlOrManifest = p === "/" || p.endsWith(".html") || p.endsWith("manifest.json");
    let bodyOut = resp.body;

    if (resp.status === 200) {
      // 确保 HTML 与静态资源始终携带 ETag，支持浏览器 If-None-Match 304 秒命中
      // 对 HTML 强制使用弱 ETag (W/"...")，避免 Cloudflare 边缘 Brotli/Gzip 压缩剥离强 ETag
      let etag = headers.get("etag");
      if (etag && isHtmlOrManifest && !/^W\//i.test(etag)) {
        etag = `W/${etag}`;
        headers.set("ETag", etag);
      }
      if (!etag && (isHtmlOrManifest || p.endsWith(".mjs") || p.endsWith(".json"))) {
        const cachedEtag = _computedEtagCache.get(p);
        if (cachedEtag) {
          etag = cachedEtag;
        } else if (request.method === "GET") {
          const ab = await resp.arrayBuffer();
          if (ab.byteLength > 0) {
            etag = await computeWeakEtagFromArrayBuffer(ab);
            _computedEtagCache.set(p, etag);
          }
          bodyOut = ab;
        } else if (request.method === "HEAD") {
          const getResp = await env.ASSETS.fetch(new Request(new URL(p, url.origin).toString(), { method: "GET" }));
          if (getResp.status === 200) {
            const ab = await getResp.arrayBuffer();
            if (ab.byteLength > 0) {
              etag = await computeWeakEtagFromArrayBuffer(ab);
              _computedEtagCache.set(p, etag);
            }
          }
        }
        if (etag) headers.set("ETag", etag);
      }

      if (p === "/sw.js") {
        headers.set("Content-Type", "application/javascript; charset=utf-8");
        headers.set("Cache-Control", "no-cache");
        headers.set("Service-Worker-Allowed", "/");
      } else if (p.endsWith(".webmanifest")) {
        headers.set("Content-Type", "application/manifest+json; charset=utf-8");
        headers.set("Cache-Control", "public, max-age=86400, stale-while-revalidate=604800");
        headers.set("CDN-Cache-Control", "public, max-age=86400, stale-while-revalidate=604800");
      } else if (p.endsWith(".gzbin") || p.endsWith(".wasm") || p.startsWith("/vendor/ort/")) {
        if (p.endsWith(".wasm")) headers.set("Content-Type", "application/wasm");
        else if (p.endsWith(".gzbin")) headers.set("Content-Type", "application/octet-stream");
        headers.set("Cache-Control", "public, max-age=31536000, immutable");
        headers.set("CDN-Cache-Control", "public, max-age=31536000, immutable");
      } else if (isVersioned && !isHtmlOrManifest) {
        // 带 ?v= 版本号的脚本/词典/韵表：永久缓存，发版改版本号即刻更新
        if (p.endsWith(".gz")) headers.set("Content-Type", "application/octet-stream");
        headers.set("Cache-Control", "public, max-age=31536000, immutable");
        headers.set("CDN-Cache-Control", "public, max-age=31536000, immutable");
      } else if (p.endsWith(".gz") || p.endsWith("/voice_prompts.json") || p.endsWith(".ico") || p.endsWith(".svg") || p.endsWith(".png")) {
        if (p.endsWith(".gz")) headers.set("Content-Type", "application/octet-stream");
        headers.set("Cache-Control", "public, max-age=86400, stale-while-revalidate=604800");
        headers.set("CDN-Cache-Control", "public, max-age=86400, stale-while-revalidate=604800");
      } else if (isHtmlOrManifest || p.endsWith(".mjs") || p.endsWith(".json")) {
        if (p === "/" || p.endsWith(".html")) {
          // 入口 HTML：5 分钟内浏览器直出本地缓存（0ms），24 小时内 stale-while-revalidate 秒开并后台校验；
          // 同时输出 Last-Modified + ETag，强刷/校验未变更返回 304 Not Modified（0 字节）
          headers.set("Last-Modified", BUILD_LAST_MODIFIED);
          headers.set("Cache-Control", "public, max-age=300, stale-while-revalidate=86400");
          headers.set("CDN-Cache-Control", "public, max-age=300, stale-while-revalidate=86400");
        } else {
          // 无版本号清单（如 manifest.json）：每次带 ETag 协商（未更新直接 304 零流量，更新立刻拿到新版本）
          headers.set("Cache-Control", "public, max-age=0, must-revalidate");
          headers.set("CDN-Cache-Control", "public, max-age=60, stale-while-revalidate=3600");
        }
      }

      // 命中 If-None-Match (ETag) 或 If-Modified-Since (Last-Modified) 立即返回 304 Not Modified（零响应体传输）
      if (request.method === "GET" || request.method === "HEAD") {
        const inm = request.headers.get("if-none-match");
        const ims = request.headers.get("if-modified-since");
        const etagHit = Boolean(etag && inm && ifNoneMatchMatches(inm, etag));
        const imsMs = ims ? Date.parse(ims) : NaN;
        const lmHit = Boolean((p === "/" || p.endsWith(".html")) && !inm && Number.isFinite(imsMs) && imsMs >= BUILD_LAST_MODIFIED_MS);
        if (etagHit || lmHit) {
          headers.delete("content-length");
          return new Response(null, { status: 304, headers });
        }
      }
    } else if (resp.status === 304) {
      if (p === "/" || p.endsWith(".html")) {
        headers.set("Last-Modified", BUILD_LAST_MODIFIED);
        headers.set("Cache-Control", "public, max-age=300, stale-while-revalidate=86400");
        headers.set("CDN-Cache-Control", "public, max-age=300, stale-while-revalidate=86400");
      } else if (isHtmlOrManifest || p.endsWith(".mjs") || p.endsWith(".json")) {
        headers.set("Cache-Control", "public, max-age=0, must-revalidate");
        headers.set("CDN-Cache-Control", "public, max-age=60, stale-while-revalidate=3600");
      }
    } else {
      headers.set("Cache-Control", "no-store");
    }

    return new Response(request.method === "HEAD" ? null : bodyOut, {
      status: resp.status,
      statusText: resp.statusText,
      headers,
    });
  },
};
