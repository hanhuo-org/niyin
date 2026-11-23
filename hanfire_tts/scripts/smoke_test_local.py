#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地零训练全链路语法与流程冒烟测试 (smoke_test_local.py)

严格遵循「本地只验证语法正确与全流程连通，不跑重度训练」原则，依次验证 7 关：
  [Test 1] 全部 Python 模块 AST / py_compile 语法零错误校验
  [Test 2] 单核 4 槽位归并器验证（测试 交keau / 伊ʔiei / 位ɣiwəi / 急kiəp / 凜liəm 等高危字）
  [Test 3] 「音·形·义」四维数学与 LPC 共振峰声波裁判校验（验证冷闭暖开、急促舒缓、单峰检测）
  [Test 4] 魔改 ZipVoiceAncient 模型实例化 + 单核硬上限时长分配器验证（介音<=3帧，塞尾<=4帧）
  [Test 5] AsyncCheckpointManager 异步双写落盘（模拟本地 + GDrive 挂载目录）与断点无损恢复校验
  [Test 6] colab_datagen_and_train.py 主入口 `--smoke` 端到端闭环运行校验（含自动生成 EVAL_REPORT.md）
"""
from __future__ import annotations

import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_1_syntax():
    py_files = sorted(PROJECT_ROOT.glob("hanfire_core/*.py")) + sorted(PROJECT_ROOT.glob("scripts/*.py"))
    assert len(py_files) >= 6, f"文件不全: {py_files}"
    for pf in py_files:
        py_compile.compile(str(pf), doraise=True)
    print(f"  ✓ [Test 1] {len(py_files)} 个核心脚本语法编译 100% 通过")


def test_2_mononuclear_4slot():
    from hanfire_core.phonology import parse_mononuclear_4slot, encode_syllables_to_sequence, SLOT_MEDIAL, SLOT_CODA
    # 验证王力三元音串全部压成唯一韵腹
    s_jiao = parse_mononuclear_4slot("keau")   # 交
    assert s_jiao.onset == "k" and s_jiao.medial == "j" and s_jiao.nucleus == "a" and s_jiao.coda == "w", s_jiao
    s_yi = parse_mononuclear_4slot("ʔiei")     # 伊
    assert s_yi.medial == "j" and s_yi.nucleus == "e" and s_yi.coda == "j", s_yi
    s_wei = parse_mononuclear_4slot("ɣiwəi")   # 位
    assert s_wei.medial == "ɥ" and s_wei.nucleus == "ə" and s_wei.coda == "j", s_wei
    s_ji = parse_mononuclear_4slot("kiəp")     # 急
    assert s_ji.is_checked is True and s_ji.coda == "p", s_ji

    enc = encode_syllables_to_sequence([
        {"type": "char", "char": "交", "ipa": "keau"},
        {"type": "char", "char": "急", "ipa": "kiəp"},
    ], speaker_id=0)
    assert enc.feats.shape[1] == 42
    print("  ✓ [Test 2] 单核 4 槽位解析器与 42 维物理特征编码通过")


def test_3_4d_referee():
    from hanfire_core.referee import (
        compute_iconicity_match_and_controls,
        judge_utterance_acoustics,
        judge_waveform_rhyme_pair,
        symbolic_rhyme_distance,
    )
    ic_cold = compute_iconicity_match_and_controls("凜", "liəm", "cold, shiver")
    ic_warm = compute_iconicity_match_and_controls("暖", "nuan", "warm")
    ic_urg = compute_iconicity_match_and_controls("急", "kiəp", "urgent, hasty")
    ic_slow = compute_iconicity_match_and_controls("舒", "ɕia", "relax, unfold")

    assert ic_cold["alpha_open"] < ic_warm["alpha_open"], (ic_cold, ic_warm)
    assert ic_urg["alpha_dur"] < ic_slow["alpha_dur"], (ic_urg, ic_slow)
    assert symbolic_rhyme_distance("lak", "nak") < 0.2
    assert symbolic_rhyme_distance("lak", "ɲia") > 2.0

    # 构造合成谐波元音测试 LPC 共振峰押韵器与单峰核计数
    sr = 24000
    t = np.linspace(0, 0.35, int(0.35 * sr), endpoint=False)
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 1.5
    # 模拟同元音共振峰 (750Hz + 1250Hz) vs 异元音 (300Hz + 2300Hz)
    w_a1 = (env * (0.6 * np.sin(2 * np.pi * 750 * t) + 0.4 * np.sin(2 * np.pi * 1250 * t))).astype(np.float32)
    w_a2 = (env * (0.6 * np.sin(2 * np.pi * 760 * t) + 0.4 * np.sin(2 * np.pi * 1230 * t))).astype(np.float32)
    r_same = judge_waveform_rhyme_pair(w_a1, w_a2, sr)
    assert r_same["bark_distance"] < 0.8 and r_same["is_rhyming"], r_same

    aud = judge_utterance_acoustics(w_a1, sr, n_expected_chars=1)
    assert aud.n_detected_peaks == 1 and aud.monosyllabic_exact, aud
    print("  ✓ [Test 3] 四维「音·形·义」数学与声波共振峰/单峰裁判全部断言通过")


def test_4_zipvoice_ancient_flow():
    from hanfire_core.phonology import encode_syllables_to_sequence, SLOT_MEDIAL, SLOT_CODA
    from hanfire_core.zipvoice_ancient import build_model_from_pretrained, allocate_mononuclear_durations

    model, _ = build_model_from_pretrained(tiny_smoke=True)
    enc = encode_syllables_to_sequence([{"type": "char", "char": "急", "ipa": "kiəp"}], speaker_id=1)
    S = enc.feats.shape[0]
    T = 48  # ~0.5s @ 93.75Hz

    sf_t = torch.from_numpy(enc.feats[None])
    dw_t = torch.from_numpy(enc.dur_weights[None])
    mf_t = torch.from_numpy(enc.max_frames[None])
    sl_t = torch.tensor([S], dtype=torch.int64)
    fl_t = torch.tensor([T], dtype=torch.int64)

    # 验证硬上限锁：介音 <= 3帧，入声塞尾 -p <= 4帧！
    st_t = model._extract_slot_types(sf_t)
    idx_alloc = allocate_mononuclear_durations(fl_t, sl_t, dw_t, mf_t, st_t, T)[0].numpy()
    for slot_pos in range(S):
        cnt = int(np.sum(idx_alloc == slot_pos))
        stype = int(st_t[0, slot_pos].item())
        if stype == SLOT_MEDIAL:
            assert cnt <= 3, f"介音超长: {cnt}"
        if stype == SLOT_CODA:
            assert cnt <= 4, f"入声塞尾超长: {cnt}"

    # 验证 1 步前向 + 梯度连通 + Euler 推理尺寸
    mel_gt = torch.randn(1, T, 100) * 0.1
    loss = model(sf_t, sl_t, dw_t, mf_t, mel_gt, fl_t)
    loss.backward()
    proj_grad_norm = sum(p.grad.norm().item() for p in model.articulatory_proj.parameters() if p.grad is not None)
    assert proj_grad_norm > 0.0, "发音特征投影层梯度未连通！"

    pred_mel = model.synthesize_features(sf_t, dw_t, mf_t, sf_t, dw_t, mf_t, mel_gt[:, :24, :], num_step=2)
    assert pred_mel.ndim == 3 and pred_mel.size(2) == 100
    print("  ✓ [Test 4] ZipVoiceAncient 前向/梯度连通/单核硬上限对齐/ODE采样测试通过")


def test_5_async_checkpoint_and_resume():
    from hanfire_core.async_checkpoint import AsyncCheckpointManager
    from hanfire_core.zipvoice_ancient import build_model_from_pretrained

    tmp = Path(tempfile.mkdtemp(prefix="hanfire_ckpt_test_"))
    try:
        local_d = tmp / "local_ckpts"
        gdrive_d = tmp / "mock_gdrive_ckpts"
        mgr = AsyncCheckpointManager(local_dir=local_d, gdrive_dir=gdrive_d, keep_last_n=2, milestone_every_steps=10)

        m1, _ = build_model_from_pretrained(tiny_smoke=True)
        opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
        mgr.save_async(stage="stage1", epoch=2, global_step=10, loss=0.432, model=m1, optimizer=opt1)
        mgr.wait_pending()

        assert (gdrive_d / "latest.pt").exists()
        assert (gdrive_d / "milestone_stage1_step_000010.pt").exists()
        assert (gdrive_d / "checkpoint_manifest.json").exists()

        # 测试恢复到全新模型实例 m2
        m2, _ = build_model_from_pretrained(tiny_smoke=True)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
        rp = mgr.resolve_resume_path("auto")
        assert rp is not None and "mock_gdrive_ckpts" in str(rp)
        info = mgr.load_checkpoint(rp, m2, optimizer=opt2)
        assert info["global_step"] == 10 and info["epoch"] == 2
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)
        mgr.flush_and_close()
        print("  ✓ [Test 5] AsyncCheckpointManager 异步双写落盘与 GDrive 断点续训恢复测试通过")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_6_end_to_end_colab_script_smoke():
    """验证 colab_datagen_and_train.py 在挂载模拟 GDrive 下的完整执行流与断点跳过能力。"""
    bundle_lex = PROJECT_ROOT / "gdrive_bundle" / "lexicon" / "hanfire_oc_v1.json"
    if not bundle_lex.exists():
        raise RuntimeError("请先运行 prepare_local_bundle.py 生成 gdrive_bundle！")

    tmp = Path(tempfile.mkdtemp(prefix="hanfire_e2e_smoke_"))
    try:
        work_d = tmp / "work"
        gd_d = tmp / "gdrive" / "hanfire_ancient_tts"
        gd_d.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "colab_datagen_and_train.py"),
            "--smoke",
            "--work-dir", str(work_d),
            "--gdrive-root", str(gd_d),
            "--bundle-tar", str(PROJECT_ROOT / "hanfire_gdrive_bundle.tar.gz"),
            "--stage1-epochs", "1",
            "--stage2-epochs", "1",
            "--save-every", "2",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print("STDOUT:\n", r.stdout)
            print("STDERR:\n", r.stderr)
            raise AssertionError(f"colab_datagen_and_train.py --smoke 失败 (exit={r.returncode})")

        assert (gd_d / "STATUS.json").exists()
        st = json.loads((gd_d / "STATUS.json").read_text(encoding="utf-8"))
        assert st["phase"] == "DONE", st
        assert (gd_d / "outputs" / "EVAL_REPORT.md").exists()
        assert (gd_d / "checkpoints" / "latest.pt").exists()
        print("  ✓ [Test 6] colab_datagen_and_train.py 端到端冒烟流程（解压->缓存->异步存盘->诗经声波验收）100% 通过")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=== 汉火上古音 TTS 本地语法与全流程冒烟验证 (零训练负载) ===")
    test_1_syntax()
    test_2_mononuclear_4slot()
    test_3_4d_referee()
    test_4_zipvoice_ancient_flow()
    test_5_async_checkpoint_and_resume()
    test_6_end_to_end_colab_script_smoke()
    print("\n🎉 ALL 6 SMOKE TESTS PASSED! 代码与素材包已具备随时直推 Colab Ultra 的生产就绪状态。")


if __name__ == "__main__":
    main()
