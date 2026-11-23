/**
 * 汉火古音工作台 PWA Service Worker (sw.js)
 * - 离线秒开 App Shell（HTML / JS / 词典 / 韵表 / 参考人声 / ORT 引擎 ≈ 8.5 MB）
 * - 严格保留 COOP / COEP / CORP 跨域隔离头，确保离线状态下多核 WASM (SharedArrayBuffer) 正常启用
 * - 不缓存 /models_chunks/*.gzbin（由 OPFS 整包零合并存储管理，避免占双份空间）
 * - 不缓存 /api/*（保证评论与打分实时）
 */
const SW_CACHE_NAME = "hanfire-pwa-20261119d";

function ensureIsolationHeaders(resp) {
  if (!resp || resp.type === "opaque") return resp;
  const h = new Headers(resp.headers);
  if (!h.has("Cross-Origin-Opener-Policy")) h.set("Cross-Origin-Opener-Policy", "same-origin");
  if (!h.has("Cross-Origin-Embedder-Policy")) h.set("Cross-Origin-Embedder-Policy", "require-corp");
  if (!h.has("Cross-Origin-Resource-Policy")) h.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(resp.body, {
    status: resp.status,
    statusText: resp.statusText,
    headers: h,
  });
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(SW_CACHE_NAME)
      .then((cache) => cache.addAll(["./", "./manifest.webmanifest"]))
      .catch(() => {})
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k.startsWith("hanfire-pwa-") && k !== SW_CACHE_NAME)
            .map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const p = url.pathname;
  // 1) API、站长后台、模型分片(.gzbin，已由 OPFS 整包管理) 与 sw.js 自身：直接放行不拦截
  if (
    p.startsWith("/api/") ||
    p === "/admin.html" ||
    p === "/sw.js" ||
    p.endsWith(".gzbin") && p.startsWith("/models_chunks/")
  ) {
    return;
  }

  // 2) 页面导航（/ 或 /index.html）：Stale-While-Revalidate —— 本机缓存 0ms 瞬开，后台静默更新
  if (req.mode === "navigate" || p === "/" || p === "/index.html") {
    event.respondWith(
      (async () => {
        const cache = await caches.open(SW_CACHE_NAME);
        const cacheKey = new Request(new URL("./", self.location.href).href);
        const cached = await cache.match(cacheKey);
        const netPromise = fetch(req)
          .then((netResp) => {
            if (netResp && netResp.ok) {
              cache.put(cacheKey, netResp.clone()).catch(() => {});
            }
            return netResp;
          })
          .catch(() => null);

        if (cached) {
          event.waitUntil(netPromise);
          return ensureIsolationHeaders(cached);
        }
        const netResp = await netPromise;
        if (netResp) return ensureIsolationHeaders(netResp);
        return new Response("当前离线且尚未缓存页面，请联网打开一次后即可永久离线使用。", {
          status: 503,
          headers: { "Content-Type": "text/plain; charset=utf-8" },
        });
      })()
    );
    return;
  }

  // 3) models_chunks/manifest.json：Stale-While-Revalidate（已缓存直接 0ms 秒回，后台静默更新，不设短超时取消）
  if (p.endsWith("/manifest.json")) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(SW_CACHE_NAME);
        const cached = await cache.match(req);
        const netPromise = fetch(req)
          .then((netResp) => {
            if (netResp && netResp.ok) {
              cache.put(req, netResp.clone()).catch(() => {});
            }
            return netResp;
          })
          .catch(() => null);
        if (cached) {
          event.waitUntil(netPromise);
          return ensureIsolationHeaders(cached);
        }
        const netResp = await netPromise;
        if (netResp) return ensureIsolationHeaders(netResp);
        return fetch(req);
      })()
    );
    return;
  }

  // 4) 静态资源（.mjs / 词典 / 韵表 / 参考人声 / ORT 引擎 / 图标 / manifest.webmanifest）：Cache-First 永久本机瞬读
  event.respondWith(
    (async () => {
      const cache = await caches.open(SW_CACHE_NAME);
      const cached = await cache.match(req);
      if (cached) {
        return ensureIsolationHeaders(cached);
      }
      const netResp = await fetch(req);
      if (netResp && netResp.ok) {
        event.waitUntil(cache.put(req, netResp.clone()).catch(() => {}));
      }
      return ensureIsolationHeaders(netResp);
    })()
  );
});
