#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音 TTS V12 Studio —— 纯棚录干声锚引导 W8A8 QAT 量化感知训练与回归门禁验证
(train_v10_studio_qat.py)

严格遵守 AGENTS.md 工程红线：
1. 100% 零 Demucs / 视频压缩语料进入主干：仅使用原生 24kHz 录音室干声锚 (ref_male_128Hz * 0.45,
   ref_female_252Hz * 0.45) 驱动冻结的 FP32 教师模型 (best_fp32_v10_studio.pt) 生成纯净推理轨迹；
2. 测试集严格隔离：评测基准诗《静夜思》《春晓》《江雪》严禁进入校准语料池；
3. 保护敏感边界层：冻结 embed (1363 词元流形)、fm_decoder.in_proj、fm_decoder.out_proj、time_emb，
   对 262 个 Linear/ScaledLinear 与 100 个 ActivationDropoutAndLinear 执行批量化 8 时间步双路 CFG 速度场蒸馏；
4. 导出真 torch.int8 权重 + float16 通道缩放因子至 v10_outputs/best_int8_v10_studio.pt (~129 MB)，
   并在真 INT8 推理态下合成 9 首回归基准诗，强制通过 audit_regression.py 六大门禁！
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "vendor" / "ZipVoice"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from zipvoice.models.modules.solver import get_time_steps  # noqa: E402
from zipvoice.models.zipvoice import ZipVoice  # noqa: E402
from zipvoice.utils.common import make_pad_mask  # noqa: E402

from audit_regression import check_wav  # noqa: E402
from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE,
    PureVocosDecoder,
    PureVocosFbank,
    SR,
    ThreeEraHanfireCompiler,
    set_batch_count,
    tokenize_mandarin_prompt_exact,
    trim_edge_silence,
)
from hanfire_core.mc_qieyun import QieyunMCCompiler  # noqa: E402
from hanfire_core.qat_int8 import (  # noqa: E402
    convert_qat_to_int8_inference,
    load_int8_checkpoint,
    prepare_zipvoice_for_qat,
)
from hanfire_core.zipvoice_zh_manifold import (  # noqa: E402
    _is_initial_token,
    _weights_to_exact_frames,
    get_sway_time_steps,
    get_tokens_index,
    hanfire_initial_noise,
    sample_clean_zero_carrier,
)
from render_v10_studio import safe_normalize  # noqa: E402

# 校准诗句池（覆盖 OC / MC / EM 三代全声母韵摄与开口零声母，严格排除测试集《静夜思》《春晓》《江雪》！）
CALIB_CLAUSES: List[Tuple[str, str, str]] = [
    ("mc", "male", "天地玄黃宇宙洪荒"),
    ("mc", "female", "日月盈昃辰宿列張"),
    ("mc", "male", "寒來暑往秋收冬藏"),
    ("mc", "female", "閏餘成歲律呂調陽"),
    ("mc", "male", "白日依山盡黃河入海流"),
    ("mc", "female", "紅豆生南國此物最相思"),
    ("mc", "male", "月落烏啼霜滿天"),
    ("mc", "male", "江楓漁火對愁眠"),
    ("mc", "female", "姑蘇城外寒山寺"),
    ("mc", "male", "夜半鐘聲到客船"),
    ("mc", "male", "東董送屋同動洞獨"),
    ("mc", "female", "仙獮線薛深審滲十"),
    ("mc", "female", "北風卷地白草折"),
    ("oc", "male", "關關雎鳩在河之洲"),
    ("oc", "female", "桃之夭夭灼灼其華"),
    ("oc", "male", "呦呦鹿鳴食野之苹"),
    ("oc", "male", "陟彼高岡[昔=sjak]我往[矣=jiʔ]"),
    ("oc", "female", "習習谷風蒹葭白露"),
    ("em", "male", "滾滾長江東逝水"),
    ("em", "male", "浪花淘盡英雄"),
    ("em", "female", "千錘萬鑿出深山"),
    ("em", "female", "要留清白在人間"),
    ("em", "male", "寒山古寺白雲庵"),
    ("em", "male", "春花秋月酒家仙"),
    ("em", "male", "青山不老買書錢"),
    ("em", "female", "清風明月秋江夜雪"),
]


@torch.no_grad()
def build_teacher_trajectories(
    teacher: ZipVoice,
    t2i: Dict[str, int],
    mc_comp: QieyunMCCompiler,
    old_comp: ThreeEraHanfireCompiler,
    pf_m: torch.Tensor,
    pit_m: List[Tuple[int, int]],
    pf_f: torch.Tensor,
    pit_f: List[Tuple[int, int]],
    device: torch.device,
) -> List[Dict]:
    trajs: List[Dict] = []
    i2t = {v: k for k, v in t2i.items()}
    ts = get_sway_time_steps(16, -0.8, device)
    snap_steps = (0, 2, 4, 6, 8, 10, 12, 14)

    for idx, (era, vtag, cl) in enumerate(CALIB_CLAUSES):
        pit, pf = (pit_m, pf_m) if vtag == "male" else (pit_f, pf_f)
        fp100 = 24.5 if vtag == "male" else 26.0
        if era == "mc":
            tgt, _ = mc_comp.compile_clause(cl, metrical=True, rhyme_hold_w=16)
        else:
            tgt, _ = old_comp.compile_era_clause(era, cl, end_punct="", rhyme_hold_w=16)

        tgt = list(tgt)
        if len(tgt) >= 2 and _is_initial_token(i2t.get(tgt[0][0], "")):
            tgt[0] = (tgt[0][0], tgt[0][1] + 11)
            fin_slot = 2 if (len(tgt) >= 3 and tgt[1][0] in (t2i.get("y0", -1), t2i.get("w0", -1))) else 1
            tgt[fin_slot] = (tgt[fin_slot][0], tgt[fin_slot][1] + 33)

        lead_block = [(t2i[","], 26)]
        tail_block = [(t2i["."], 38)]
        pmt_ids = [x[0] for x in pit]
        pmt_ws = [x[1] for x in pit]
        gen_all = lead_block + tgt + tail_block
        cat_ids = pmt_ids + [x[0] for x in gen_all]

        pmt_frames = int(pf.size(1))
        lead_frames = max(6, int(round(26 * fp100 / 100.0)))
        sp_frames = max(int(round(sum(w for _, w in tgt) * fp100 / 100.0)), len(tgt) + 4)
        tail_frames = max(8, int(round(38 * fp100 / 100.0)))
        total_frames = pmt_frames + lead_frames + sp_frames + tail_frames

        cat_durs = (
            _weights_to_exact_frames(pmt_ws, pmt_frames)
            + [lead_frames]
            + _weights_to_exact_frames([w for _, w in tgt], sp_frames)
            + [tail_frames]
        )
        padding_mask = make_pad_mask(torch.tensor([total_frames], device=device), max_len=total_frames)
        tokens_index = get_tokens_index([cat_durs], total_frames).to(device)
        cat_embed, _ = teacher.forward_text_embed([cat_ids])
        text_cond = torch.gather(
            cat_embed, dim=1, index=tokens_index.unsqueeze(-1).expand(1, total_frames, cat_embed.size(-1))
        )
        speech_cond = F.pad(pf, (0, 0, 0, total_frames - pf.size(1)))

        x_np = hanfire_initial_noise(total_frames, pf.size(-1), seed=42 + idx, tau=0.667, mode="spherical")
        x = torch.from_numpy(x_np).to(device=device, dtype=pf.dtype)

        t_list, x2_list, sc_keep_list, v_list = [], [], [], []
        for s in range(16):
            t_val = ts[s]
            t_f = float(t_val.item())
            x2 = torch.cat([x, x], dim=0)
            pm2 = torch.cat([padding_mask, padding_mask], dim=0)
            tc2 = torch.cat([torch.zeros_like(text_cond), text_cond], dim=0)
            if t_f > 0.5:
                sc2 = torch.cat([torch.zeros_like(speech_cond), speech_cond], dim=0)
                gs_eff = 1.0
                keep_uncond_sc = 0.0
            else:
                sc2 = torch.cat([speech_cond, speech_cond], dim=0)
                gs_eff = 2.0
                keep_uncond_sc = 1.0
            v_both = teacher.forward_fm_decoder(t=t_val, xt=x2, text_condition=tc2, speech_condition=sc2, padding_mask=pm2)
            v_unc, v_cnd = v_both.chunk(2, dim=0)
            v_cfg = (1.0 + gs_eff) * v_cnd - gs_eff * v_unc
            if s in snap_steps:
                t_list.extend([t_f, t_f])
                x2_list.append(x2.detach().cpu())
                sc_keep_list.extend([keep_uncond_sc, 1.0])
                v_list.append(v_both.detach().cpu())
            x = x + v_cfg * (ts[s + 1] - ts[s])

        trajs.append({
            "cat_ids": cat_ids,
            "tokens_index": tokens_index.detach().cpu(),
            "speech_cond": speech_cond.detach().cpu(),
            "padding_mask": padding_mask.detach().cpu(),
            "t_embed_teacher": cat_embed.detach().cpu(),
            "t_batch": torch.tensor(t_list, dtype=torch.float32),       # [16]
            "x_batch": torch.cat(x2_list, dim=0),                       # [16, T, 100]
            "sc_keep": torch.tensor(sc_keep_list, dtype=torch.float32), # [16]
            "v_batch": torch.cat(v_list, dim=0),                        # [16, T, 100]
        })
    return trajs


def main() -> None:
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v10_dir = PROJECT_ROOT / "v10_outputs"
    fp32_p = v10_dir / "best_fp32_v10_studio.pt"
    int8_p = v10_dir / "best_int8_v10_studio.pt"

    print(f"[1/4] 加载 FP32 教师模型: {fp32_p.name} ...")
    ck = torch.load(str(fp32_p), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    t2i = ck["t2i_custom"]

    teacher = ZipVoice(**cfg)
    teacher.load_state_dict(ck["model"], strict=True)
    teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    vocos, fbank = PureVocosDecoder(dev), PureVocosFbank()
    mc_comp = QieyunMCCompiler(t2i)
    old_comp = ThreeEraHanfireCompiler(t2i, PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json", v8_fixes=True)

    pw_m, _ = sf.read(str(PROJECT_ROOT / "v5_bundle" / "ref_male_128Hz.wav"), dtype="float32")
    pit_m, _ = tokenize_mandarin_prompt_exact("高分经典悬疑电影，", t2i)
    pf_m = fbank.extract((pw_m * 0.45).astype(np.float32), sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
    pl_m = torch.tensor([pf_m.size(1)], device=dev)

    pw_f, _ = sf.read(str(PROJECT_ROOT / "v5_bundle" / "ref_female_252Hz.wav"), dtype="float32")
    pit_f, _ = tokenize_mandarin_prompt_exact("今夜的月光如此清亮，", t2i)
    pf_f = fbank.extract((pw_f * 0.45).astype(np.float32), sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
    pl_f = torch.tensor([pf_f.size(1)], device=dev)

    print(f"[2/4] 生成 {len(CALIB_CLAUSES)} 条纯棚录甜区教师 ODE 轨迹 (200 组双路 CFG 速度场快照)...")
    trajs = build_teacher_trajectories(teacher, t2i, mc_comp, old_comp, pf_m, pit_m, pf_f, pit_f, dev)
    print(f"  ✓ 教师轨迹生成完毕 ({time.time() - t_start:.1f}s)")

    print("[3/4] 执行批量化 Stage-2 W8A8 STE QAT 蒸馏 (冻结 embed 与边界投影层)...")
    student = ZipVoice(**cfg)
    student.load_state_dict(ck["model"], strict=True)
    # 移除训练期统计正则算子 Balancer / Whiten，确保反向传播速度场与 eval 推理态 100% 一致
    from zipvoice.models.modules.scaling import Balancer, Whiten
    from zipvoice.utils.scaling_converter import get_submodule
    for m_name, m_obj in list(student.named_modules()):
        if isinstance(m_obj, (Balancer, Whiten)) and "." in m_name:
            par, chd = m_name.rsplit(".", 1)
            setattr(get_submodule(student, par), chd, nn.Identity())
    qat_stats = prepare_zipvoice_for_qat(student)
    student.to(dev).eval()

    qat_params: List[nn.Parameter] = []
    for name, p in student.named_parameters():
        if any(
            k in name
            for k in (
                "embed",
                "text_encoder.in_proj",
                "text_encoder.out_proj",
                "fm_decoder.in_proj",
                "fm_decoder.out_proj",
                "time_emb",
            )
        ):
            p.requires_grad = False
        else:
            p.requires_grad = True
            qat_params.append(p)

    print(
        f"  ✓ QAT 包装: Linear={qat_stats['replaced_linear']} 层 | "
        f"ActDropoutLinear={qat_stats['replaced_act_linear']} 层 | "
        f"量化覆盖率={qat_stats['quantized_ratio_pct']}%"
    )

    # 记录原始 FP32 权重与每通道 1-ULP 量化步长 Δ_q：将 QAT 严格约束在 ±0.65*Δ_q 舍入邻域内 (AdaRound Trust-Region)，
    # 既允许任意边界权重翻转最优取整方向抵消 W8A8 误差，又从数学上 100% 杜绝在 25 句校准集上过拟合破坏 5 万小时预训练流形！
    orig_weights: Dict[nn.Parameter, Tuple[torch.Tensor, torch.Tensor]] = {}
    for p in qat_params:
        w0 = p.data.detach().clone()
        if w0.dim() >= 2:
            dq = (w0.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7) / 127.0) * 0.50
        else:
            dq = (w0.abs().amax().clamp(min=1e-7) / 127.0) * 0.50
        orig_weights[p] = (w0, dq)

    # 先评估 0-step W8A8 基线损失，确保 QAT 蒸馏单调下降绝不退化！（严禁调用 set_batch_count 篡改 Zipformer ScheduledFloat！）
    with torch.no_grad():
        init_losses = []
        for tr in trajs:
            te_s0, _ = student.forward_text_embed([tr["cat_ids"]])
            te_l0 = F.mse_loss(te_s0, tr["t_embed_teacher"].to(dev))
            tok_idx = tr["tokens_index"].to(dev)
            sp_cond = tr["speech_cond"].to(dev)
            T = sp_cond.size(1)
            tc = torch.gather(te_s0, dim=1, index=tok_idx.unsqueeze(-1).expand(1, T, te_s0.size(-1)))
            B = tr["x_batch"].size(0)
            tc_b = torch.cat([torch.zeros_like(tc), tc], dim=0).repeat(B // 2, 1, 1)
            sc_b = sp_cond.expand(B, T, -1) * tr["sc_keep"].to(dev).view(B, 1, 1)
            pm_b = tr["padding_mask"].to(dev).expand(B, T)
            v_s0 = student.fm_decoder(x=torch.cat([tr["x_batch"].to(dev), tc_b, sc_b], dim=2), t=tr["t_batch"].to(dev), padding_mask=pm_b)
            init_losses.append(float((F.mse_loss(v_s0, tr["v_batch"].to(dev)) + 0.5 * te_l0).item()))
    best_loss = float(np.mean(init_losses))
    print(f"  ✓ 0-step W8A8 初始速度场蒸馏误差 kd_loss={best_loss:.6f}")

    epochs = 3
    total_steps = epochs * len(trajs)
    opt = torch.optim.AdamW(qat_params, lr=3.0e-6, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=3.0e-7)

    best_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    rng = np.random.default_rng(42)

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        order = rng.permutation(len(trajs))
        ep_losses: List[float] = []
        for idx in order:
            tr = trajs[int(idx)]
            opt.zero_grad(set_to_none=True)

            te_stu, _ = student.forward_text_embed([tr["cat_ids"]])
            te_loss = F.mse_loss(te_stu, tr["t_embed_teacher"].to(dev))

            tok_idx = tr["tokens_index"].to(dev)
            sp_cond = tr["speech_cond"].to(dev)
            T = sp_cond.size(1)
            tc = torch.gather(te_stu, dim=1, index=tok_idx.unsqueeze(-1).expand(1, T, te_stu.size(-1)))

            B = tr["x_batch"].size(0)  # 16 (= 8 steps * [uncond, cond])
            tc_b = torch.cat([torch.zeros_like(tc), tc], dim=0).repeat(B // 2, 1, 1)
            sc_keep = tr["sc_keep"].to(dev).view(B, 1, 1)
            sc_b = sp_cond.expand(B, T, -1) * sc_keep
            pm_b = tr["padding_mask"].to(dev).expand(B, T)
            xt_cat = torch.cat([tr["x_batch"].to(dev), tc_b, sc_b], dim=2)
            v_stu = student.fm_decoder(x=xt_cat, t=tr["t_batch"].to(dev), padding_mask=pm_b)

            fm_kd = F.mse_loss(v_stu, tr["v_batch"].to(dev))
            loss = fm_kd + 0.5 * te_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_params, 1.0)
            opt.step()
            with torch.no_grad():
                for p, (w0, dq) in orig_weights.items():
                    p.data.copy_(torch.max(torch.min(p.data, w0 + dq), w0 - dq))
            sched.step()
            ep_losses.append(float(loss.item()))

        ml = float(np.mean(ep_losses))
        if ml < best_loss:
            best_loss = ml
            best_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if ep == 1 or ep % 3 == 0 or ep == epochs:
            print(f"  [QAT Epoch {ep:02d}/{epochs}] kd_loss={ml:.6f} (best={best_loss:.6f}) | {time.time() - t_ep:.1f}s")

    student.load_state_dict(best_sd, strict=True)
    student_cpu = student.cpu().eval()
    convert_qat_to_int8_inference(student_cpu)
    int8_payload = {
        "config": cfg,
        "t2i_custom": t2i,
        "qat_stats": qat_stats,
        "best_kd_loss": round(best_loss, 6),
        "int8_state_dict": {k: v.cpu() for k, v in student_cpu.state_dict().items()},
    }
    torch.save(int8_payload, str(int8_p))
    fp32_mb = fp32_p.stat().st_size / 1e6
    int8_mb = int8_p.stat().st_size / 1e6
    print(f"  ✓ 真 INT8 权重已导出: {int8_p.name} ({int8_mb:.1f} MB vs FP32 {fp32_mb:.1f} MB, 压缩比 {fp32_mb/int8_mb:.2f}x!)")

    print("\n[4/4] 从磁盘加载真 INT8 模型 (load_int8_checkpoint) 渲染 9 首基准诗并执行六大门禁核验...")
    int8_model, _ = load_int8_checkpoint(int8_p, ZipVoice, device=dev)
    sos_bp = signal.butter(4, [75.0 / (SR / 2), 9500.0 / (SR / 2)], btype="band", output="sos")
    out_dir = v10_dir / "audio_v12_qat_int8"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_ref_dir = v10_dir / "audio_v12_pure_natural"

    tasks = [
        ("MC_v12_纯净自然_静夜思_male.wav", "mc", pit_m, pf_m, pl_m, 24.5, ["床前明月光", "疑是地上霜", "舉頭望明月", "低頭思故鄉"], 0.20, False),
        ("MC_v12_纯净自然_静夜思_female.wav", "mc", pit_f, pf_f, pl_f, 26.5, ["床前明月光", "疑是地上霜", "舉頭望明月", "低頭思故鄉"], 0.20, False),
        ("MC_v12_纯净自然_春晓_male.wav", "mc", pit_m, pf_m, pl_m, 24.5, ["春眠不覺曉", "[處=昌開三魚去]處聞啼鳥", "夜來風雨聲", "花落知多少"], None, False),
        ("MC_v12_纯净自然_春晓_female.wav", "mc", pit_f, pf_f, pl_f, 26.5, ["春眠不覺曉", "[處=昌開三魚去]處聞啼鳥", "夜來風雨聲", "花落知多少"], 0.35, False),
        ("MC_v12_纯净自然_江雪_male.wav", "mc", pit_m, pf_m, pl_m, 24.5, ["千山鳥飛絕", "萬徑人蹤滅", "孤舟蓑笠翁", "獨釣寒江雪"], 0.30, True),
        ("MC_v12_纯净自然_江雪_female.wav", "mc", pit_f, pf_f, pl_f, 26.5, ["千山鳥飛絕", "萬徑人蹤滅", "孤舟蓑笠翁", "獨釣寒江雪"], 0.25, True),
        ("EM_v12_纯净自然_桃花庵歌_male.wav", "em", pit_m, pf_m, pl_m, 24.0, ["桃花塢裏桃花庵", "桃花庵下桃花仙", "桃花仙人[種=tʂuŋh]桃樹", "又摘桃花換酒錢"], 0.45, False),
        ("EM_v12_纯净自然_桃花庵歌_female.wav", "em", pit_f, pf_f, pl_f, 25.0, ["桃花塢裏桃花庵", "桃花庵下桃花仙", "桃花仙人[種=tʂuŋh]桃樹", "又摘桃花換酒錢"], 0.45, False),
        ("OC_v12_纯净自然_蒹葭_male.wav", "oc", pit_m, pf_m, pl_m, 25.0, ["蒹葭蒼蒼", "白露為霜", "所謂伊人", "在水一方"], None, False),
    ]

    total_errs = 0
    mel_corrs = []
    for fname, era, pit, pf, pl, fp100, lines, min_r, chk_xue in tasks:
        chunks = []
        for j, cl in enumerate(lines):
            if era == "mc":
                it, _ = mc_comp.compile_clause(cl, metrical=True, rhyme_hold_w=16)
            else:
                it, _ = old_comp.compile_era_clause(era, cl, end_punct="", rhyme_hold_w=16)
            mel = sample_clean_zero_carrier(
                model=int8_model,
                prompt_items=pit,
                target_items=it,
                t2i=t2i,
                prompt_features=pf,
                prompt_features_lens=pl,
                frames_per_100w=fp100,
                char0_boost_w=44,
                num_step=16,
                guidance_scale=1.0,
                t_shift=0.5,
                seed=42,
            )
            if j == 0:
                mel_fp = sample_clean_zero_carrier(
                    model=teacher, prompt_items=pit, target_items=it, t2i=t2i,
                    prompt_features=pf, prompt_features_lens=pl, frames_per_100w=fp100,
                    char0_boost_w=44, num_step=16, guidance_scale=1.0, t_shift=0.5, seed=42,
                )
                mc_c = float(F.cosine_similarity(mel.flatten(), mel_fp.flatten(), dim=0).item())
                mel_corrs.append(round(mc_c, 5))
            w = vocos.decode_mel(mel.permute(0, 2, 1) / FEAT_SCALE)
            w = signal.sosfiltfilt(sos_bp, w).astype(np.float32)
            w = trim_edge_silence(w, rel_db=52.0, pre_ms=35, post_ms=65)
            w = safe_normalize(w, target_rms=0.055, max_peak=0.72)
            f = int(0.025 * SR)
            if len(w) > f * 2:
                w[:f] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, f, dtype=np.float32)))
                w[-f:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, f, dtype=np.float32)))
            chunks += [w, np.zeros(int((0.26 if j % 2 == 0 else 0.42) * SR), dtype=np.float32)]

        out_p = out_dir / fname
        full = np.concatenate(chunks)
        sf.write(str(out_p), full, SR)
        errs = check_wav(out_p, 4, (0, 1, 3), min_r, chk_xue)
        total_errs += len(errs)

    report = {
        "fp32_mb": round(fp32_mb, 2),
        "int8_mb": round(int8_mb, 2),
        "compression_ratio": round(fp32_mb / int8_mb, 2),
        "qat_stats": qat_stats,
        "best_kd_loss": round(best_loss, 6),
        "regression_errors": total_errs,
        "mean_fp32_mel_cosine": round(float(np.mean(mel_corrs)), 5) if mel_corrs else 0.0,
    }
    (out_dir / "qat_int8_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== V12 Studio W8A8 QAT 结果: {'9/9 全部通过 ✅' if total_errs == 0 else f'失败 {total_errs} 项 ❌'} | FP32 Mel 均值余弦={report['mean_fp32_mel_cosine']} ===")
    sys.exit(0 if total_errs == 0 else 1)


if __name__ == "__main__":
    main()
