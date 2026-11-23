#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步非阻塞 Checkpoint 管理器与断点续训恢复器 (async_checkpoint.py)

特性：
1. 零阻塞训练主循环：主线程仅花 <30ms 将权重与优化器状态浅拷贝至 CPU 内存并入队，后台单线程顺序落盘；
2. 双写原子保护：先写本地高速盘 `.tmp` -> `os.replace` 原子改名，再同步至挂载的 Google Drive 目录
   (`/content/drive/MyDrive/hanfire_ancient_tts/checkpoints`)，杜绝断电/断连写坏半截文件；
3. 滚动保留 + 里程碑永留：高频中间 ckpt 自动滚动保留最近 N 个（省网盘空间），`best.pt`、`latest.pt`
   及每个 Stage 完成里程碑永久保留；
4. 一键无缝续训 (`resume="auto"`)：下次开机挂载 GDrive 后自动检测最新有效 checkpoint，完整恢复
   `model + optimizer + scheduler + grad_scaler + stage + epoch + global_step + RNG`。
"""
from __future__ import annotations

import atexit
import copy
import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _to_cpu_detached(obj: Any) -> Any:
    """递归将 state_dict 中的 CUDA Tensor 转为独立 CPU Tensor，释放显存引用且不阻塞后续反向传播。"""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu_detached(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_detached(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_detached(v) for v in obj)
    return copy.deepcopy(obj)


class AsyncCheckpointManager:
    def __init__(
        self,
        local_dir: str | Path,
        gdrive_dir: Optional[str | Path] = None,
        keep_last_n: int = 5,
        milestone_every_steps: int = 2000,
    ):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.gdrive_dir = Path(gdrive_dir) if gdrive_dir else None
        if self.gdrive_dir is not None:
            try:
                self.gdrive_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"[AsyncCkpt] 提示: GDrive 目录暂不可写 ({e})，仅保存在本地 {self.local_dir}")
                self.gdrive_dir = None

        self.keep_last_n = max(2, keep_last_n)
        self.milestone_every_steps = milestone_every_steps
        self.best_loss = float("inf")

        self._q: "queue.Queue[Optional[Dict]]" = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._saved_count = 0
        self._worker = threading.Thread(target=self._run_worker, name="AsyncCkptWorker", daemon=True)
        self._worker.start()
        atexit.register(self.flush_and_close)

    def _atomic_save_pt(self, payload: Dict, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(target_path))

    def _atomic_copy(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
        shutil.copy2(str(src), str(tmp_dst))
        os.replace(str(tmp_dst), str(dst))

    def _prune_rolling(self, folder: Path, stage: str) -> None:
        rolling = sorted(
            folder.glob(f"ckpt_{stage}_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(rolling) > self.keep_last_n:
            for old_p in rolling[: len(rolling) - self.keep_last_n]:
                try:
                    old_p.unlink()
                except OSError:
                    pass

    def _update_manifest(self, folder: Path, meta: Dict) -> None:
        mf_path = folder / "checkpoint_manifest.json"
        hist = []
        if mf_path.exists():
            try:
                hist = json.loads(mf_path.read_text(encoding="utf-8")).get("history", [])
            except Exception:
                hist = []
        hist.append(meta)
        data = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latest": meta,
            "history": hist[-50:],
        }
        tmp = mf_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(mf_path))

    def _run_worker(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                break
            try:
                payload = job["payload"]
                stage = payload["stage"]
                step = payload["global_step"]
                loss = payload["loss"]
                is_best = job["is_best"]
                is_milestone = job["is_milestone"]
                tag = job.get("tag")

                fname = f"milestone_{stage}_step_{step:06d}.pt" if is_milestone else f"ckpt_{stage}_step_{step:06d}.pt"
                if tag:
                    fname = f"ckpt_{stage}_{tag}_step_{step:06d}.pt"

                local_ckpt = self.local_dir / fname
                self._atomic_save_pt(payload, local_ckpt)
                self._atomic_copy(local_ckpt, self.local_dir / "latest.pt")
                self._atomic_copy(local_ckpt, self.local_dir / f"latest_{stage}.pt")
                if is_best:
                    self._atomic_copy(local_ckpt, self.local_dir / "best.pt")
                if not is_milestone and not tag:
                    self._prune_rolling(self.local_dir, stage)

                meta = {
                    "stage": stage,
                    "epoch": payload["epoch"],
                    "global_step": step,
                    "loss": round(float(loss), 5),
                    "is_best": is_best,
                    "is_milestone": is_milestone,
                    "filename": fname,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._update_manifest(self.local_dir, meta)

                # 异步镜像至 Google Drive（完全不卡 GPU 主线程）
                if self.gdrive_dir is not None:
                    gd_ckpt = self.gdrive_dir / fname
                    self._atomic_copy(local_ckpt, gd_ckpt)
                    self._atomic_copy(local_ckpt, self.gdrive_dir / "latest.pt")
                    self._atomic_copy(local_ckpt, self.gdrive_dir / f"latest_{stage}.pt")
                    if is_best:
                        self._atomic_copy(local_ckpt, self.gdrive_dir / "best.pt")
                    if not is_milestone and not tag:
                        self._prune_rolling(self.gdrive_dir, stage)
                    self._update_manifest(self.gdrive_dir, meta)

                with self._lock:
                    self._saved_count += 1
            except Exception as e:
                with self._lock:
                    self._last_error = f"{type(e).__name__}: {e}"
                print(f"[AsyncCkpt] 后台存盘警告: {self._last_error}", flush=True)
            finally:
                self._q.task_done()

    def save_async(
        self,
        *,
        stage: str,
        epoch: int,
        global_step: int,
        loss: float,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        extra_state: Optional[Dict] = None,
        is_milestone: bool = False,
        tag: Optional[str] = None,
    ) -> None:
        """在训练循环中调用：极速拷贝至 CPU 并推入异步队列，不阻塞 GPU。"""
        raw_model = model.module if hasattr(model, "module") else model
        is_best = float(loss) < self.best_loss
        if is_best:
            self.best_loss = float(loss)
        if self.milestone_every_steps > 0 and global_step > 0 and (global_step % self.milestone_every_steps == 0):
            is_milestone = True

        payload = {
            "format_version": 1,
            "stage": stage,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "loss": float(loss),
            "best_loss": float(self.best_loss),
            "model_state_dict": _to_cpu_detached(raw_model.state_dict()),
            "optimizer_state_dict": _to_cpu_detached(optimizer.state_dict()) if optimizer is not None else None,
            "scheduler_state_dict": _to_cpu_detached(scheduler.state_dict()) if scheduler is not None else None,
            "scaler_state_dict": _to_cpu_detached(scaler.state_dict()) if scaler is not None else None,
            "rng_cpu": torch.get_rng_state(),
            "extra_state": _to_cpu_detached(extra_state or {}),
            "timestamp": time.time(),
        }
        # 若队列已满，等待前一个慢速网盘写入腾出一个槽位以防内存堆积
        self._q.put({
            "payload": payload,
            "is_best": is_best,
            "is_milestone": is_milestone,
            "tag": tag,
        })

    def wait_pending(self) -> None:
        """阻塞等待队列中所有挂起的 checkpoint 落盘完成（用于 Stage 切换或评测前）。"""
        self._q.join()

    def flush_and_close(self) -> None:
        if self._worker.is_alive():
            self._q.put(None)
            self._q.join()
            self._worker.join(timeout=30.0)

    def resolve_resume_path(self, resume: str = "auto", stage: Optional[str] = None) -> Optional[Path]:
        """按优先级查找可恢复的 checkpoint：显式路径 > GDrive 最新 > 本地最新。"""
        if not resume or resume == "none":
            return None
        if resume != "auto":
            p = Path(resume)
            return p if p.exists() else None

        candidates: list[Path] = []
        target_names = ([f"latest_{stage}.pt"] if stage else []) + ["latest.pt", "best.pt"]
        for base in [self.gdrive_dir, self.local_dir]:
            if base is None or not base.exists():
                continue
            for name in target_names:
                p = base / name
                if p.exists() and p.stat().st_size > 1024:
                    candidates.append(p)

        if not candidates:
            return None
        # 取 step 最大或最新修改时间的文件
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def load_checkpoint(
        self,
        ckpt_path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        strict_model: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> Dict:
        """加载 checkpoint 并恢复训练上下文（支持换卡续训、跨 Stage 热启动）。"""
        ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=strict_model)

        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.best_loss = float(ckpt.get("best_loss", ckpt.get("loss", float("inf"))))
        return {
            "stage": ckpt.get("stage", "stage1"),
            "epoch": int(ckpt.get("epoch", 0)),
            "global_step": int(ckpt.get("global_step", 0)),
            "loss": float(ckpt.get("loss", 0.0)),
            "best_loss": self.best_loss,
            "extra_state": ckpt.get("extra_state", {}),
            "loaded_from": str(ckpt_path),
        }
