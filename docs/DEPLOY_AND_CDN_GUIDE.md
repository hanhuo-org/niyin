# 汉火古音工作台（`yin.hanhuo.org`）部署、优选 IP、PWA 与缓存运维指南

本文档记录 `https://yin.hanhuo.org/` 线上最终稳定版（已实测秒开 + 离线可用）的 **Cloudflare 单域名优选 IP 架构**、**PWA 离线缓存与 OPFS 模型零合并存储体系**、**发版刷新版本号清单** 与 **一键验收/测速脚本**，方便后续发版、换 IP 或排错时直接复用。

---

## 一、核心网络架构（单域名全 CF 灰云 `A` 记录方案）

当前 `hanhuo.org` 注册于 Spaceship，NS 托管在 Cloudflare（`kianchau.ns.cloudflare.com` / `kim.ns.cloudflare.com`）。采用**单域名免 SaaS 方案**，无需第二个域名、无需迁 DNSPod：

1. **Worker 触发器使用经典 `Route`（严禁用 `custom_domain = true`）**
   - [wrangler.toml](./wrangler.toml)：
     ```toml
     routes = [
       { pattern = "yin.hanhuo.org/*", zone_name = "hanhuo.org" }
     ]
     ```
   - 原因：`custom_domain = true` 会生成带锁的 Proxied（小黄云）`Worker` DNS 记录，无法自选 IP；使用 `Route` 后，只要流量经由 Cloudflare Anycast 节点到达并携带 `Host: yin.hanhuo.org`，Worker 路由与边缘 HTTPS 证书自动生效。

2. **Cloudflare DNS 使用「双灰云 `A` 记录」（零 IPv6 + 零 CNAME 嵌套）**
   - 线上当前生效配置（`hanhuo.org` → **DNS → Records**）：
     | Type | Name | IPv4 address | Proxy status |
     | :---: | :---: | :--- | :---: |
     | `A` | `yin` | **`104.16.78.156`** | **关闭（DNS only / 灰云）** |
     | `A` | `yin` | **`104.18.25.40`** | **关闭（DNS only / 灰云）** |
   - **为什么最终选灰云 `A` 记录而不是 CNAME？**
     - 彻底杜绝任何 DNS / DoH 返回慢速跨境 IPv6（`AAAA = []`）；
     - 省去 2~3 层嵌套 CNAME 查询；
     - 两个不同网段（`104.16.x.x` + `104.18.x.x`）均为 Cloudflare 永久自有的全球 Anycast 核心段，自带双活容错，不会“停用下线”；若未来路由变慢，10 秒替换新 IP 即可。

---

## 二、已实测可用优选 IP 池与避坑清单

> ⚠️ **单域名 Worker Route 选 IP 铁律**：
> - **绝对禁止**选用 Cloudflare DNS 名称服务器段（IPv4 `108.162.x.x` / `173.245.58.x` / `172.64.32.x`；IPv6 `2606:4700:50::` / `2a06:98c1:50::` / `2803:f800:50::`），直连会返回 **`HTTP 403 (error code: 1034)`** 或导致手机/家宽 IPv6 优先连接卡死重试！

### 1. 已验证 100% `HTTP 200` 的纯净优选 IPv4（可直接填灰云 `A` 记录）
以下 10 个 IP 均已针对 `https://yin.hanhuo.org/`（首页 + `manifest.webmanifest` + `719 KB` 词典）实测通过，零 `1034`：
- **线上主用**：`104.16.78.156`、`104.18.25.40`
- **纯净备用**：`104.18.25.69`、`104.18.25.16`、`103.31.4.47`、`104.17.157.37`、`104.18.42.54`、`104.21.22.53`、`104.20.33.190`、`104.26.10.221`
- **无 IPv6 大厂 CNAME 备选**：`shopify.com`（`23.227.38.33`，`AAAA = 0`）

### 2. 已踩坑不可用/不推荐的 CNAME
- ❌ `yin.cloudflare.byoip.top` / `*.cnae.top`：底层 `cnae.top` 失效，境内 DNS 返回 `NXDOMAIN`。
- ❌ `yin.cloudflare.19931110.xyz` / `yin.cloudflare.22336666.xyz`：40 个 IP 中混入 19 个 `108.162.198.x`（如 `108.162.198.89`），近一半请求报 `403 error code: 1034`。
- ❌ `yin.cloudflare-dualstack.19931110.xyz` / `*.cloudflare-dualstack.byoip.top`：自带 21 条 `:50::` DNS 段毒 IPv6，开启 IPv6 的宽带/5G 会卡死重试。
- ⚠️ `cf.877774.xyz`：在非联通/海外 DoH 下会回落 `www.wto.org` 并返回慢速 IPv6（`2a06:98c1:3100::...`），建议直接取其纯 IPv4 填 `A` 记录。
- ⚠️ `www.visa.com.sg`：虽无 IPv6 且 100% 200，但晚高峰部分国内宽带限速严重（低至 ~5 KB/s）。

---

## 三、首屏秒开、PWA 离线与 OPFS 零合并存储架构

### 1. 四层缓存体系（首次极简加载 `~0.8 MB`，二次 `~5 ms` 离线秒开）
1. **PWA Service Worker 离线壳（[sw.js](./sw.js) + [manifest.webmanifest](./manifest.webmanifest)）**：
   - 离线缓存页面 HTML（Stale-While-Revalidate 秒开 + 后台静默更新）、3 个 `.mjs`、三代词典（`719 KB`）、韵表（`77 KB`）、参考人声（`277 KB`）、ORT 引擎（`7.0 MB`）与图标；
   - **严格保留 `COOP / COEP / CORP` 响应头**，确保离线/飞行模式下依然跑满多核 WASM（`SharedArrayBuffer`）；
   - **不缓存 `/models_chunks/*.gzbin`**（交由 OPFS 管理，零双份磁盘占用），**不缓存 `/api/*`**；
   - `models_chunks/manifest.json` 采用 Stale-While-Revalidate，确保离线也能识别本机已缓存模型；
   - 桌面 Chrome/Edge 及安卓浏览器支持顶部「📲 安装应用」一键安装到桌面（含 `icon-192.png` / `icon-512.png`），iOS Safari 支持「分享 → 添加到主屏幕」全屏运行。
2. **OPFS 完整模型零合并存储（[model_cache.mjs](./model_cache.mjs)）**：
   - **全端按需下载**：无论电脑还是手机，打开网页后**不自动下模型、不拉 ORT 引擎**，点「开始古音拟音合成」或「▶ 听本韵」时再按需下载/唤醒（手机端仍先弹流量提醒）；
   - 首次下载分片落盘 IndexedDB（保留断网续传）；全部拼装校验完成后，自动将 3 个完整 `.onnx` 写入浏览器原生 **OPFS（`hanfire_models_v1`）**，并**立即删除 IndexedDB 冗余分片**（硬盘仅占 1 份，且规避 iOS IndexedDB 大对象 OOM）；
   - 下次打开点击合成时，经 Worker `createSyncAccessHandle` 一次 I/O **整包零合并直读**，并与 `ORT WASM` 流水线并行初始化；旧浏览器自动回落为**单次只读事务并发读取 IndexedDB**。
3. **首屏零阻塞（[index.html](./index.html) + [hanfire_worker.mjs](./hanfire_worker.mjs)）**：
   - `#synthBtn` 首帧即亮可点（不再初始灰置）；
   - `inspect_env` 放到 `window.load` 后启动（消除浏览器标签页持续转圈），内部「词典加载 + WebGPU 探测（2s 超时兜底）+ OPFS/IDB 扫描」三路并行；
   - 词典拉取带完整正文超时保护（25s 超时自动重连，防慢网假死）；
   - 全韵表（`rhyme_catalog.json`）、公开评论墙（`/api/feedback/public`）、参考人声（`voice_prompts.json.gz`，从 921 KB 压至 277 KB）、ORT 引擎均按需懒加载；移除 47 KB `favicon.ico`，仅保留 1.9 KB `favicon.svg`。
4. **Cloudflare 边缘 HTTP 缓存策略（[cf_worker.mjs](./cf_worker.mjs)）**：
   - 带 `?v=` 版本号的脚本/词典/韵表/参考人声、`.gzbin`、`.wasm`、`/vendor/ort/*`：`ETag` + `Cache-Control` & `CDN-Cache-Control: public, max-age=31536000, immutable`；
   - 入口 HTML（`/`、`*.html`）：`Last-Modified`（`BUILD_LAST_MODIFIED`）+ `Cache-Control: public, max-age=300, stale-while-revalidate=86400`，支持 `If-Modified-Since → 304`（规避 CF 边缘对 `text/html` 压缩/邮箱混淆剥离 ETag）；
   - `manifest.json`：`ETag` + `max-age=0, must-revalidate` + `If-None-Match → 304`；
   - `/sw.js`：`Cache-Control: no-cache` + `Service-Worker-Allowed: /`；
   - `/api/feedback/public`：`public, max-age=15, stale-while-revalidate=60`；非 `2xx` 一律 `no-store`。

---

## 四、日常发版与刷新缓存标准流程

### 1. 发版改版本号清单（改完即平滑更新，无旧缓存残留）
- 若修改了 `.mjs` / 词典 / 韵表 / 参考人声：同步把 [index.html](./index.html) 与 [hanfire_worker.mjs](./hanfire_worker.mjs) 里的 `ASSET_VER`（当前 `20261119d`）递增（如 `20261120a`）；
- 若修改了 `index.html`：把 [cf_worker.mjs](./cf_worker.mjs) 里的 `BUILD_LAST_MODIFIED` 更新为当前 GMT 时间；
- 若修改了 `sw.js` 缓存结构（仅必要时，否则保持 `SW_CACHE_NAME` 不变以保留用户已缓存的 719 KB 词典）：更新 [sw.js](./sw.js) 里的 `SW_CACHE_NAME`。

### 2. 同步静态目录并部署到 Cloudflare Workers
```bash
# 1. 同步改动文件到 dist_cf/
cp /mnt/s790/project/tts/hanfire_tts/web_wasm/{index.html,hanfire_worker.mjs,model_cache.mjs,sw.js,manifest.webmanifest} \
   /mnt/s790/project/tts/hanfire_tts/web_wasm/dist_cf/

# 2. 确认 dist_cf/ 中没有本地临时试听目录（如有 _tang_review / tang_review.html 先临时移走，部署后再移回）
ls -la /mnt/s790/project/tts/hanfire_tts/web_wasm/dist_cf/

# 3. 部署到 Cloudflare Workers（不会影响 Cloudflare DNS 中的灰云 A 记录）
cd /mnt/s790/project/tts/hanfire_tts/web_wasm
CI=1 /mnt/s790/cache/mise-installs/wrangler/4.143.0/node_modules/.bin/wrangler deploy
```

---

## 五、一键验收与换 IP 测速脚本

### 1. 线上全链路验收（DNS + 纯 IPv4 + 304 + PWA + D1 API）
```bash
python3 - <<'PY'
import urllib.request, json, subprocess

a_j = json.loads(urllib.request.urlopen("https://dns.alidns.com/resolve?name=yin.hanhuo.org&type=A", timeout=5).read().decode())
aaaa_j = json.loads(urllib.request.urlopen("https://dns.alidns.com/resolve?name=yin.hanhuo.org&type=AAAA", timeout=5).read().decode())
ips = [a["data"] for a in a_j.get("Answer", []) if a["type"] == 1]
v6 = [a["data"] for a in aaaa_j.get("Answer", []) if a["type"] == 28]
print(f"AliDNS: A={ips}  AAAA={v6} (期望 AAAA=[])")

for ip in ips:
    for path in ["/", "/sw.js", "/manifest.webmanifest", "/api/feedback/public?limit=1"]:
        flag = "-s" if path.startswith("/api/") else "-sI"
        out = subprocess.check_output(
            ["curl", flag, "--connect-timeout", "5", "--resolve", f"yin.hanhuo.org:443:{ip}",
             f"https://yin.hanhuo.org{path}"], text=True
        )
        first = out.splitlines()[0] if out else ""
        print(f"  [{ip}] {path:28s} -> {first[:80]}")
PY
```

### 2. 本机直连宽带快速挑最快 IP（关代理后运行）
```bash
for ip in 104.16.78.156 104.18.25.40 104.18.25.69 104.18.25.16 103.31.4.47 104.17.157.37 104.18.42.54 104.21.22.53 104.20.33.190 104.26.10.221; do
  curl -s -o /dev/null --connect-timeout 4 -m 8 \
    -w "$ip -> HTTP=%{http_code} tcp=%{time_connect}s tls=%{time_appconnect}s ttfb=%{time_starttransfer}s\n" \
    --resolve yin.hanhuo.org:443:$ip https://yin.hanhuo.org/manifest.webmanifest
done
```

---

## 六、手机端 CSS 改版防回归要点

后续改动 [index.html](./index.html) 样式时，请守住两条实测踩过的移动端 CSS 红线：
1. **防 Grid 被横滑条撑爆视口**：`.grid` 轨道须用 `minmax(0, 1fr)`，且 `.grid > *`、`.card`、`.presets`、`.main-nav` 须保持 `min-width: 0; max-width: 100%;`，防止单行横滑预设条把 `390px` 视口撑宽到 `1100px+`。
2. **防 `.era-btn` 嵌套术语折行**：对副标题样式必须使用直接子代选择器 `.era-btn > span`，不可写成 `.era-btn span`（否则会把内部 `<span class="term">` 全部变成块级元素换行）。
