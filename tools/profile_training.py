"""
Training bottleneck profiler for cross-encoder pipeline.

Run with:
    accelerate launch --num_processes=8 profile_training.py
    # or single GPU:
    python profile_training.py

Reports time spent in: sampler build, data loading, H2D transfer,
tokenization, forward, backward, optimizer, all-reduce, and idle gaps.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("DISABLE_TQDM", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("CONSOLE_LOG_LEVEL", "INFO")

import torch
import torch.distributed as dist
import numpy as np

# ─── distributed helpers ─────────────────────────────────────────────────────

def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    try:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except ValueError:
        rank = 0
    try:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        ws = 1
    return rank, ws

def _is_main() -> bool:
    rank, _ = _rank_world()
    return rank == 0

def _local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return 0

# ─── GPU utilization background poller ───────────────────────────────────────

class GpuUtilPoller:
    """Polls nvidia-smi every interval_s on the main process."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=3).decode()
                ts = time.time()
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 6:
                        continue
                    try:
                        self.samples.append({
                            "ts": ts,
                            "gpu": int(parts[0]),
                            "gpu_util": float(parts[1]),
                            "mem_util": float(parts[2]),
                            "mem_used_mb": float(parts[3]),
                            "mem_total_mb": float(parts[4]),
                            "power_w": float(parts[5]) if parts[5] not in ("N/A", "[N/A]") else float("nan"),
                        })
                    except ValueError:
                        pass
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        by_gpu: dict[int, list[dict]] = defaultdict(list)
        for s in self.samples:
            by_gpu[s["gpu"]].append(s)
        result = {}
        for gpu_idx, rows in sorted(by_gpu.items()):
            utils = [r["gpu_util"] for r in rows]
            mem_utils = [r["mem_util"] for r in rows]
            powers = [r["power_w"] for r in rows if not np.isnan(r["power_w"])]
            result[f"gpu{gpu_idx}"] = {
                "gpu_util_mean": statistics.mean(utils),
                "gpu_util_p10": sorted(utils)[len(utils) // 10],
                "gpu_util_p90": sorted(utils)[int(len(utils) * 0.9)],
                "mem_util_mean": statistics.mean(mem_utils),
                "power_w_mean": statistics.mean(powers) if powers else float("nan"),
                "n_samples": len(rows),
            }
        return result


# ─── CUDA event timer ─────────────────────────────────────────────────────────

class CudaTimer:
    """Accumulates GPU-side elapsed time via CUDA events."""

    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        self._samples_ms: list[float] = []
        self._start: torch.cuda.Event | None = None

    def start(self):
        if self.device.type == "cuda":
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()

    def stop(self):
        if self.device.type == "cuda" and self._start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            self._samples_ms.append(self._start.elapsed_time(end))
            self._start = None

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self._samples_ms) if self._samples_ms else 0.0

    @property
    def total_ms(self) -> float:
        return sum(self._samples_ms)

    @property
    def n(self) -> int:
        return len(self._samples_ms)


# ─── wall-clock step timer ────────────────────────────────────────────────────

class WallTimer:
    def __init__(self, name: str):
        self.name = name
        self._samples: list[float] = []
        self._t0: float | None = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        if self._t0 is not None:
            self._samples.append(time.perf_counter() - self._t0)
            self._t0 = None

    @property
    def mean_s(self) -> float:
        return statistics.mean(self._samples) if self._samples else 0.0

    @property
    def total_s(self) -> float:
        return sum(self._samples)

    @property
    def n(self) -> int:
        return len(self._samples)


# ─── synthetic data helpers ───────────────────────────────────────────────────

def _make_fake_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Fake tokenized batch: {input_ids, attention_mask, token_type_ids}."""
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    ttype = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": ttype}


def _fake_text_pairs(batch_size: int, query_len: int = 64, doc_len: int = 256) -> list[list[str]]:
    """Pairs of random ASCII strings to simulate tokenizer input."""
    def rstr(n):
        import random, string
        return " ".join("".join(random.choices(string.ascii_lowercase, k=6)) for _ in range(n // 6))
    return [[rstr(query_len), rstr(doc_len)] for _ in range(batch_size)]


# ─── Section 1: Sampler build timing ─────────────────────────────────────────

def profile_sampler_build(
    train_dataset_path: str,
    rank: int,
    world_size: int,
    batch_size: int,
    no_duplicates: bool,
    bucket_assignment: str = "greedy",
) -> dict[str, float]:
    print(f"\n{'='*60}")
    print("SECTION 1: Sampler / Dataset Build Timing")
    print(f"{'='*60}")

    results: dict[str, float] = {}

    # Import from training script location
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from sampler_bucketed_fixed import make_same_dataset_batch_sampler
        print("  Using sampler_bucketed_fixed")
    except ImportError:
        try:
            from sampler_bucketed import make_same_dataset_batch_sampler
            print("  Using sampler_bucketed")
        except ImportError:
            from sampler import make_same_dataset_batch_sampler
            print("  Using sampler (basic)")

    from PositiveOnly_train_bucketed import (
        load_bucketed_split,
        load_plain_parquets_with_groups,
        build_key_maps,
        stable_hash63,
    )

    # Dataset load
    t0 = time.perf_counter()
    if (Path(train_dataset_path) / "manifest.json").exists() or \
       (Path(train_dataset_path).parent / "manifest.json").exists():
        dataset, groups = load_bucketed_split(
            train_dataset_path, split="train",
            rank=rank, world_size=world_size, assignment=bucket_assignment,
        )
    else:
        dataset, groups = load_plain_parquets_with_groups(train_dataset_path)
    dataset_load_s = time.perf_counter() - t0
    results["dataset_load_s"] = dataset_load_s
    print(f"  dataset load:       {dataset_load_s:.2f}s  (rows={len(dataset)}, groups={len(groups)})")

    # Key map build
    t0 = time.perf_counter()
    index_to_anchor, index_to_positive_key = build_key_maps(dataset, no_duplicates=no_duplicates)
    keymap_s = time.perf_counter() - t0
    results["keymap_build_s"] = keymap_s
    print(f"  key map build:      {keymap_s:.2f}s  (no_duplicates={no_duplicates})")

    # Sampler factory creation
    t0 = time.perf_counter()
    sampler_factory = make_same_dataset_batch_sampler(
        groups,
        no_duplicates=no_duplicates,
        index_to_anchor=index_to_anchor,
        index_to_positive_key=index_to_positive_key,
        repeat_for_accelerate_sharding=False,
    )
    factory_s = time.perf_counter() - t0
    results["sampler_factory_s"] = factory_s
    print(f"  sampler factory:    {factory_s:.3f}s")

    # Instantiate one sampler (what happens per epoch) to measure __iter__ build phase
    import torch as _torch
    gen = _torch.Generator()
    gen.manual_seed(42)
    sampler = sampler_factory(dataset, batch_size=batch_size, drop_last=True, generator=gen, seed=42)
    t0 = time.perf_counter()
    # Consume a few batches to trigger the full build (all_reduce + schedule)
    it = iter(sampler)
    first_batch = next(it)
    iter_first_s = time.perf_counter() - t0
    results["sampler_first_batch_s"] = iter_first_s
    print(f"  sampler first batch (full build + first yield): {iter_first_s:.2f}s")
    print(f"  schedule length: {getattr(sampler, '_last_schedule_len', 'unknown')}")

    del dataset
    gc.collect()
    return results


# ─── Section 2: Dataloader throughput (with real tokenizer) ──────────────────

def profile_dataloader_throughput(
    train_dataset_path: str,
    model_name_or_path: str,
    rank: int,
    world_size: int,
    batch_size: int,
    max_length: int,
    num_workers: int,
    no_duplicates: bool,
    n_batches: int = 50,
) -> dict[str, float]:
    print(f"\n{'='*60}")
    print(f"SECTION 2: Dataloader Throughput  (num_workers={num_workers})")
    print(f"{'='*60}")

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from sampler_bucketed_fixed import make_same_dataset_batch_sampler
    except ImportError:
        try:
            from sampler_bucketed import make_same_dataset_batch_sampler
        except ImportError:
            from sampler import make_same_dataset_batch_sampler

    from PositiveOnly_train_bucketed import (
        load_bucketed_split,
        load_plain_parquets_with_groups,
        build_key_maps,
        sanitize_training_dataset_for_loss,
    )
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def collate_fn(batch):
        pairs = []
        for row in batch:
            query = row.get("query") or row.get("anchor") or ""
            positive = row.get("positive") or ""
            if isinstance(positive, list):
                positive = positive[0] if positive else ""
            pairs.append([str(query), str(positive)])
        tok = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tok

    if (Path(train_dataset_path) / "manifest.json").exists() or \
       (Path(train_dataset_path).parent / "manifest.json").exists():
        dataset, groups = load_bucketed_split(
            train_dataset_path, split="train",
            rank=rank, world_size=world_size, assignment="greedy",
        )
    else:
        dataset, groups = load_plain_parquets_with_groups(train_dataset_path)

    index_to_anchor, index_to_positive_key = build_key_maps(dataset, no_duplicates=no_duplicates)
    dataset = sanitize_training_dataset_for_loss(dataset)

    sampler_factory = make_same_dataset_batch_sampler(
        groups,
        no_duplicates=no_duplicates,
        index_to_anchor=index_to_anchor,
        index_to_positive_key=index_to_positive_key,
    )
    gen = torch.Generator()
    gen.manual_seed(42)
    batch_sampler = sampler_factory(dataset, batch_size=batch_size, drop_last=True, generator=gen, seed=42)

    dl = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )

    # Warm up + measure
    intervals: list[float] = []
    tok_shapes: list[tuple] = []
    t_prev = time.perf_counter()
    for i, batch in enumerate(dl):
        t_now = time.perf_counter()
        intervals.append(t_now - t_prev)
        tok_shapes.append(tuple(batch["input_ids"].shape))
        t_prev = t_now
        if i >= n_batches:
            break

    warmup = 3
    measured = intervals[warmup:]
    results = {
        "mean_batch_interval_s": statistics.mean(measured) if measured else 0,
        "p90_batch_interval_s": sorted(measured)[int(len(measured) * 0.9)] if measured else 0,
        "throughput_samples_per_s": batch_size / statistics.mean(measured) if measured else 0,
        "typical_seq_len": tok_shapes[-1][1] if tok_shapes else 0,
    }

    print(f"  mean batch interval: {results['mean_batch_interval_s']*1000:.1f} ms")
    print(f"  p90  batch interval: {results['p90_batch_interval_s']*1000:.1f} ms")
    print(f"  throughput:          {results['throughput_samples_per_s']:.1f} samples/s")
    print(f"  typical seq_len:     {results['typical_seq_len']}")

    del dataset
    gc.collect()
    return results


# ─── Section 3: Pure GPU throughput (synthetic) ───────────────────────────────

def profile_gpu_throughput(
    model_name_or_path: str,
    batch_size: int,
    mini_batch_size: int,
    max_length: int,
    n_steps: int = 30,
    loss_scale: float = 10.0,
) -> dict[str, float]:
    print(f"\n{'='*60}")
    print("SECTION 3: GPU Throughput (synthetic batches, no IO)")
    print(f"{'='*60}")

    device = torch.device(f"cuda:{_local_rank()}" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
    from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss
    from transformers import AutoTokenizer

    print("  Loading model...")
    t0 = time.perf_counter()
    model = CrossEncoder(
        model_name_or_path,
        max_length=max_length,
        model_card_data=CrossEncoderModelCardData(language="ru", license="apache-2.0"),
        model_kwargs={"attn_implementation": "flash_attention_2"},
    )
    model_load_s = time.perf_counter() - t0
    print(f"  model load: {model_load_s:.2f}s")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base_model = model.model.to(device)

    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5, fused=True)

    def tokenize_pairs(pairs: list[list[str]]) -> dict[str, torch.Tensor]:
        enc = tokenizer(
            pairs, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    # Generate fake text pairs (simulate real tokenization cost)
    print("  Generating synthetic text pairs...")
    fake_pairs = _fake_text_pairs(batch_size)

    timers = {
        "tokenize": WallTimer("tokenize"),
        "h2d": WallTimer("h2d"),
        "forward_mini": WallTimer("forward_mini"),
        "backward": WallTimer("backward"),
        "optimizer": WallTimer("optimizer"),
    }
    cuda_fwd = CudaTimer("cuda_forward", device)
    cuda_bwd = CudaTimer("cuda_backward", device)

    n_mini = max(1, batch_size // mini_batch_size)
    print(f"  batch_size={batch_size}, mini_batch_size={mini_batch_size}, n_mini={n_mini}")

    base_model.train()
    for step in range(n_steps + 3):  # 3 warmup
        # Tokenize (simulates collate_fn, happens on CPU)
        timers["tokenize"].start()
        enc = tokenize_pairs(fake_pairs)
        timers["tokenize"].stop()

        # Forward in mini-batches (CachedMNRL pattern)
        optimizer.zero_grad(set_to_none=True)
        all_logits = []

        cuda_fwd.start()
        timers["forward_mini"].start()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for mi in range(n_mini):
                s = mi * mini_batch_size
                e = s + mini_batch_size
                mini_enc = {k: v[s:e] for k, v in enc.items()}
                logits = base_model(**mini_enc).logits
                all_logits.append(logits)
        timers["forward_mini"].stop()
        cuda_fwd.stop()

        # Aggregate and backward
        cuda_bwd.start()
        timers["backward"].start()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_cat = torch.cat(all_logits, dim=0).squeeze(-1)
            labels = torch.arange(batch_size, dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits_cat.unsqueeze(0).expand(batch_size, -1) * loss_scale, labels)
        loss.backward()
        timers["backward"].stop()
        cuda_bwd.stop()

        timers["optimizer"].start()
        optimizer.step()
        timers["optimizer"].stop()

    results = {}
    for name, t in timers.items():
        if t.n > 3:
            samples = t._samples[3:]  # skip warmup
            results[f"{name}_mean_ms"] = statistics.mean(samples) * 1000
    results["cuda_forward_mean_ms"] = cuda_fwd.mean_ms
    results["cuda_backward_mean_ms"] = cuda_bwd.mean_ms
    total_step_ms = sum(
        results.get(f"{k}_mean_ms", 0)
        for k in ("tokenize", "h2d", "forward_mini", "backward", "optimizer")
    )
    results["total_step_est_ms"] = total_step_ms
    results["model_load_s"] = model_load_s

    print(f"\n  Per-step breakdown (mean over {n_steps} steps):")
    print(f"    tokenize:           {results.get('tokenize_mean_ms', 0):.1f} ms  (CPU)")
    print(f"    forward (wall):     {results.get('forward_mini_mean_ms', 0):.1f} ms")
    print(f"    forward (CUDA):     {results.get('cuda_forward_mean_ms', 0):.1f} ms")
    print(f"    backward (wall):    {results.get('backward_mean_ms', 0):.1f} ms")
    print(f"    backward (CUDA):    {results.get('cuda_backward_mean_ms', 0):.1f} ms")
    print(f"    optimizer:          {results.get('optimizer_mean_ms', 0):.1f} ms")
    print(f"    total (estimated):  {total_step_ms:.1f} ms  → {1000/total_step_ms:.1f} steps/s")

    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ─── Section 4: Tokenization micro-benchmark ──────────────────────────────────

def profile_tokenizer(
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    n_trials: int = 20,
) -> dict[str, float]:
    print(f"\n{'='*60}")
    print("SECTION 4: Tokenizer Throughput")
    print(f"{'='*60}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    short_pairs = _fake_text_pairs(batch_size, query_len=32, doc_len=64)
    medium_pairs = _fake_text_pairs(batch_size, query_len=64, doc_len=256)
    long_pairs = _fake_text_pairs(batch_size, query_len=128, doc_len=512)

    results = {}
    for label, pairs in [("short(~100tok)", short_pairs), ("medium(~320tok)", medium_pairs), ("long(~640tok)", long_pairs)]:
        times = []
        for _ in range(n_trials + 2):
            t0 = time.perf_counter()
            enc = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            times.append(time.perf_counter() - t0)
        measured = times[2:]
        mean_ms = statistics.mean(measured) * 1000
        seq_len = enc["input_ids"].shape[1]
        results[f"tokenize_{label}_ms"] = mean_ms
        print(f"  {label:20s}: {mean_ms:.1f} ms  (seq_len={seq_len})  →  {batch_size/statistics.mean(measured):.0f} samples/s")

    return results


# ─── Section 5: AllReduce bandwidth ──────────────────────────────────────────

def profile_allreduce(world_size: int) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized() and world_size > 1):
        print("\n  AllReduce: skipped (single process)")
        return {}

    print(f"\n{'='*60}")
    print("SECTION 5: AllReduce Bandwidth / Latency")
    print(f"{'='*60}")

    device = torch.device(f"cuda:{_local_rank()}" if torch.cuda.is_available() else "cpu")
    results = {}

    for size_mb in [0.001, 1, 10, 100]:
        n_floats = int(size_mb * 1024 * 1024 / 4)
        t = torch.randn(n_floats, device=device)
        # Warmup
        for _ in range(3):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        mean_ms = statistics.mean(times) * 1000
        bw_gbs = size_mb / 1024 / statistics.mean(times)
        results[f"allreduce_{size_mb}mb_ms"] = mean_ms
        print(f"  all_reduce {size_mb:6.3f} MB: {mean_ms:.2f} ms  →  {bw_gbs:.2f} GB/s")

    return results


# ─── Section 6: End-to-end step timing with real dataloader ──────────────────

def profile_e2e_steps(
    train_dataset_path: str,
    model_name_or_path: str,
    rank: int,
    world_size: int,
    batch_size: int,
    mini_batch_size: int,
    max_length: int,
    num_workers: int,
    no_duplicates: bool,
    n_steps: int = 30,
    loss_scale: float = 10.0,
) -> dict[str, float]:
    print(f"\n{'='*60}")
    print("SECTION 6: End-to-End Step Profiling")
    print(f"  (real data, real model, num_workers={num_workers})")
    print(f"{'='*60}")

    device = torch.device(f"cuda:{_local_rank()}" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from sampler_bucketed_fixed import make_same_dataset_batch_sampler
    except ImportError:
        try:
            from sampler_bucketed import make_same_dataset_batch_sampler
        except ImportError:
            from sampler import make_same_dataset_batch_sampler

    from PositiveOnly_train_bucketed import (
        load_bucketed_split,
        load_plain_parquets_with_groups,
        build_key_maps,
        sanitize_training_dataset_for_loss,
    )
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = CrossEncoder(
        model_name_or_path, max_length=max_length,
        model_card_data=CrossEncoderModelCardData(language="ru", license="apache-2.0"),
        model_kwargs={"attn_implementation": "flash_attention_2"},
    )
    base_model = model.model.to(device)

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model, device_ids=[_local_rank()], output_device=_local_rank()
        )

    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5, fused=True)

    if (Path(train_dataset_path) / "manifest.json").exists() or \
       (Path(train_dataset_path).parent / "manifest.json").exists():
        dataset, groups = load_bucketed_split(
            train_dataset_path, split="train",
            rank=rank, world_size=world_size, assignment="greedy",
        )
    else:
        dataset, groups = load_plain_parquets_with_groups(train_dataset_path)

    index_to_anchor, index_to_positive_key = build_key_maps(dataset, no_duplicates=no_duplicates)
    dataset = sanitize_training_dataset_for_loss(dataset)

    sampler_factory = make_same_dataset_batch_sampler(
        groups, no_duplicates=no_duplicates,
        index_to_anchor=index_to_anchor, index_to_positive_key=index_to_positive_key,
    )
    gen = torch.Generator()
    gen.manual_seed(42 + rank)
    batch_sampler = sampler_factory(dataset, batch_size=batch_size, drop_last=True, generator=gen, seed=42)

    def collate_fn(rows):
        pairs = []
        for row in rows:
            q = row.get("query") or row.get("anchor") or ""
            p = row.get("positive") or ""
            if isinstance(p, list):
                p = p[0] if p else ""
            pairs.append([str(q), str(p)])
        return tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    dl = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )

    # Timing buckets
    t_data_fetch: list[float] = []
    t_h2d: list[float] = []
    t_forward: list[float] = []
    t_backward: list[float] = []
    t_optimizer: list[float] = []
    t_step_total: list[float] = []
    seq_lens: list[int] = []

    dl_iter = iter(dl)
    warmup = 3
    base_model.train()

    for step in range(n_steps + warmup):
        t_step_start = time.perf_counter()

        # Data fetch (includes tokenization in collate_fn + worker overhead)
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch = next(dl_iter)
        t_data = time.perf_counter() - t0

        # H2D
        t0 = time.perf_counter()
        batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_h2d_val = time.perf_counter() - t0

        seq_len = batch["input_ids"].shape[1]

        # Forward
        optimizer.zero_grad(set_to_none=True)
        n_mini = max(1, batch_size // mini_batch_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            all_logits = []
            for mi in range(n_mini):
                s = mi * mini_batch_size
                e = s + mini_batch_size
                mini = {k: v[s:e] for k, v in batch_gpu.items()}
                logits = base_model(**mini).logits
                all_logits.append(logits)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        # Backward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_cat = torch.cat(all_logits, dim=0).squeeze(-1)
            labels = torch.arange(logits_cat.shape[0], device=device)
            loss = torch.nn.functional.cross_entropy(
                logits_cat.unsqueeze(0).expand(logits_cat.shape[0], -1) * loss_scale,
                labels,
            )
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t0

        # Optimizer
        t0 = time.perf_counter()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_opt = time.perf_counter() - t0

        t_step = time.perf_counter() - t_step_start

        if step >= warmup:
            t_data_fetch.append(t_data)
            t_h2d.append(t_h2d_val)
            t_forward.append(t_fwd)
            t_backward.append(t_bwd)
            t_optimizer.append(t_opt)
            t_step_total.append(t_step)
            seq_lens.append(seq_len)

    def _ms(lst):
        return statistics.mean(lst) * 1000 if lst else 0.0

    results = {
        "data_fetch_mean_ms": _ms(t_data_fetch),
        "h2d_mean_ms": _ms(t_h2d),
        "forward_mean_ms": _ms(t_forward),
        "backward_mean_ms": _ms(t_backward),
        "optimizer_mean_ms": _ms(t_optimizer),
        "step_total_mean_ms": _ms(t_step_total),
        "seq_len_mean": statistics.mean(seq_lens) if seq_lens else 0,
        "gpu_idle_frac": _ms(t_data_fetch) / _ms(t_step_total) if t_step_total else 0,
    }

    compute_ms = _ms(t_forward) + _ms(t_backward) + _ms(t_optimizer)
    print(f"\n  Per-step breakdown ({n_steps} steps, seq_len≈{results['seq_len_mean']:.0f}):")
    print(f"    data fetch + tok:   {results['data_fetch_mean_ms']:.1f} ms  ← GPU IDLE during this")
    print(f"    H2D transfer:       {results['h2d_mean_ms']:.1f} ms")
    print(f"    forward ({n_mini} mini): {results['forward_mean_ms']:.1f} ms")
    print(f"    backward:           {results['backward_mean_ms']:.1f} ms")
    print(f"    optimizer:          {results['optimizer_mean_ms']:.1f} ms")
    print(f"    TOTAL step:         {results['step_total_mean_ms']:.1f} ms")
    print(f"    GPU idle fraction:  {results['gpu_idle_frac']*100:.1f}%  (data_fetch / total)")
    print(f"    Compute fraction:   {compute_ms / _ms(t_step_total) * 100:.1f}%")

    del dataset, base_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ─── Report ──────────────────────────────────────────────────────────────────

def _print_report(all_results: dict[str, Any]):
    rank, _ = _rank_world()
    if rank != 0:
        return

    print(f"\n{'='*60}")
    print("BOTTLENECK SUMMARY")
    print(f"{'='*60}")

    # GPU utilization
    gpu = all_results.get("gpu_util", {})
    if gpu:
        for gname, stats in gpu.items():
            util = stats["gpu_util_mean"]
            flag = " ← LOW UTILIZATION" if util < 60 else ""
            print(f"  {gname}: GPU util {util:.1f}%  mem_util {stats['mem_util_mean']:.1f}%  power {stats['power_w_mean']:.0f}W{flag}")

    # Sampler
    s = all_results.get("sampler", {})
    if s:
        total_startup = s.get("dataset_load_s", 0) + s.get("keymap_build_s", 0) + s.get("sampler_first_batch_s", 0)
        print(f"\n  Startup overhead per epoch (rank 0):")
        print(f"    dataset load:   {s.get('dataset_load_s', 0):.1f}s")
        print(f"    keymap build:   {s.get('keymap_build_s', 0):.1f}s")
        print(f"    sampler build:  {s.get('sampler_first_batch_s', 0):.1f}s")
        print(f"    TOTAL startup:  {total_startup:.1f}s  ← multiplied by num_epochs")

    # E2E step
    e = all_results.get("e2e", {})
    if e:
        data_ms = e.get("data_fetch_mean_ms", 0)
        total_ms = e.get("step_total_mean_ms", 0)
        idle_pct = e.get("gpu_idle_frac", 0) * 100
        print(f"\n  Training step bottleneck:")
        if idle_pct > 30:
            print(f"    *** DATA LOADING is the bottleneck ({idle_pct:.0f}% GPU idle) ***")
            print(f"        Increase dataloader_num_workers (currently 0)")
        compute_ms = e.get("forward_mean_ms", 0) + e.get("backward_mean_ms", 0)
        if compute_ms < 100 and total_ms > 200:
            print(f"    *** Compute time ({compute_ms:.0f}ms) << step time ({total_ms:.0f}ms) → IO bound ***")

    print(f"\n  Top recommendations:")
    print(f"  1. Set dataloader_num_workers=4 (or 8). Currently 0 → all tokenization blocks GPU.")
    print(f"  2. Set dataloader_pin_memory=True when num_workers>0.")
    print(f"  3. Pre-tokenize and cache the dataset as input_ids tensors (eliminates tokenization from hot path).")
    print(f"  4. If no_duplicates=True: consider pre-building batches once offline with prepare_query_buckets.py.")
    print(f"  5. Set mini_batch_size=batch_size to use 1 forward pass instead of N serial mini-batches.")
    print(f"     (only do this if VRAM allows)")
    print(f"  6. Use gradient_checkpointing only if OOM; it trades ~30% compute for memory.")
    print(f"  7. Check if flash_attention_2 is actually being used (model card / logs).")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-encoder training bottleneck profiler")
    parser.add_argument("--model", default="/home/jovyan/new-pvc/cross_encoders/nik_datasets/models/RuModernBERT-base")
    parser.add_argument("--train-data", default="/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/final_dataset_bucketed_v5/train/")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mini-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers to test (0=current config)")
    parser.add_argument("--no-duplicates", action="store_true", default=True)
    parser.add_argument("--n-steps", type=int, default=30, help="steps for e2e profiling")
    parser.add_argument("--loss-scale", type=float, default=10.0)
    parser.add_argument("--skip-sampler", action="store_true", help="skip sampler build profiling (slow)")
    parser.add_argument("--skip-e2e", action="store_true", help="skip end-to-end step profiling (slow)")
    parser.add_argument("--skip-gpu-profile", action="store_true", help="skip synthetic GPU throughput test")
    parser.add_argument("--output-json", default="profile_results.json")

    args = parser.parse_args()

    # Init distributed if launched via accelerate
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not (dist.is_available() and dist.is_initialized()):
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank, world_size = _rank_world()

    if _is_main():
        print(f"\n{'#'*60}")
        print("# Cross-Encoder Training Bottleneck Profiler")
        print(f"# rank={rank}  world_size={world_size}")
        print(f"# model={args.model}")
        print(f"# train_data={args.train_data}")
        print(f"# batch_size={args.batch_size}  mini_batch_size={args.mini_batch_size}")
        print(f"# max_length={args.max_length}  num_workers={args.num_workers}")
        print(f"{'#'*60}\n")

    # Start GPU poller on main process
    poller = None
    if _is_main() and torch.cuda.is_available():
        poller = GpuUtilPoller(interval_s=0.5).start()

    all_results: dict[str, Any] = {}

    # Section 1: Sampler build (only rank 0 for simplicity)
    if not args.skip_sampler and rank == 0:
        try:
            all_results["sampler"] = profile_sampler_build(
                train_dataset_path=args.train_data,
                rank=rank,
                world_size=world_size,
                batch_size=args.batch_size,
                no_duplicates=args.no_duplicates,
            )
        except Exception as e:
            print(f"  [Section 1 FAILED]: {e}")

    # Section 4: Tokenizer (fast, always run)
    if rank == 0:
        try:
            all_results["tokenizer"] = profile_tokenizer(
                model_name_or_path=args.model,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        except Exception as e:
            print(f"  [Section 4 FAILED]: {e}")

    # Section 3: Pure GPU throughput
    if not args.skip_gpu_profile and torch.cuda.is_available():
        try:
            all_results["gpu_throughput"] = profile_gpu_throughput(
                model_name_or_path=args.model,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                max_length=args.max_length,
                n_steps=args.n_steps,
                loss_scale=args.loss_scale,
            )
        except Exception as e:
            print(f"  [Section 3 FAILED]: {e}")

    # Section 5: AllReduce
    try:
        all_results["allreduce"] = profile_allreduce(world_size)
    except Exception as e:
        print(f"  [Section 5 FAILED]: {e}")

    # Section 6: E2E steps with real data
    if not args.skip_e2e:
        try:
            all_results["e2e"] = profile_e2e_steps(
                train_dataset_path=args.train_data,
                model_name_or_path=args.model,
                rank=rank,
                world_size=world_size,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                max_length=args.max_length,
                num_workers=args.num_workers,
                no_duplicates=args.no_duplicates,
                n_steps=args.n_steps,
                loss_scale=args.loss_scale,
            )
        except Exception as e:
            print(f"  [Section 6 FAILED]: {e}")

    # Stop GPU poller + collect
    if poller is not None:
        time.sleep(1)  # let last samples arrive
        poller.stop()
        all_results["gpu_util"] = poller.summary()
        print(f"\n{'='*60}")
        print("GPU Utilization (sampled during profiling)")
        print(f"{'='*60}")
        for gname, stats in all_results["gpu_util"].items():
            print(f"  {gname}: util={stats['gpu_util_mean']:.1f}%  mem_util={stats['mem_util_mean']:.1f}%  "
                  f"power={stats['power_w_mean']:.0f}W  samples={stats['n_samples']}")

    # Print summary on rank 0
    if _is_main():
        _print_report(all_results)

        out = Path(args.output_json)
        out.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"\nFull results written to: {out.resolve()}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
