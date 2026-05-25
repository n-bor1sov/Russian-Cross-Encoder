
import gc
import hashlib
import json
import logging
import logging.handlers
import os
import queue
import re
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["HF_HOME"] = os.path.expanduser("/home/jovyan/new-pvc/cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("/home/jovyan/new-pvc/cache/huggingface/datasets")
# Live progress bars write frequently to stdout/stderr and can block under
# launchers/ClearML/tee when the pipe fills. Disable by default; enable only
# for short interactive runs with ENABLE_TQDM=1.
os.environ.setdefault("DISABLE_TQDM", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
# Keep stdout/stderr quiet when not overridden (avoids pipe stalls under tee/kubectl).
os.environ.setdefault("CONSOLE_LOG_LEVEL", "ERROR")

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from clearml import Task
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from torch.utils.data import ConcatDataset, DataLoader
from sampler import make_same_dataset_batch_sampler
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from transformers import TrainerCallback

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
_LOG_LISTENERS: list[logging.handlers.QueueListener] = []
_WATCHDOG_THREADS: dict[int, threading.Thread] = {}


class UnshardedCrossEncoderTrainer(CrossEncoderTrainer):
    """CrossEncoderTrainer variant that leaves a rank-local batch sampler intact.

    SentenceTransformers 5.2.2 always calls ``accelerator.prepare`` on the train
    dataloader, which wraps custom batch samplers in ``BatchSamplerShard``. This
    job already loads rank-local data and synchronizes the sampler schedule
    itself, so the dataloader must not be sharded a second time.
    """

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Training requires specifying a train_dataset to the SentenceTransformerTrainer.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if isinstance(train_dataset, IterableDataset):
            dataloader_params.update(
                {
                    "batch_size": self.args.train_batch_size,
                    "drop_last": self.args.dataloader_drop_last,
                }
            )
        elif isinstance(train_dataset, IterableDatasetDict):
            raise ValueError(
                "Sentence Transformers is not compatible with IterableDatasetDict. Please use a DatasetDict instead."
            )
        elif isinstance(train_dataset, DatasetDict):
            for dataset in train_dataset.values():
                if isinstance(dataset, IterableDataset):
                    raise ValueError(
                        "Sentence Transformers is not compatible with a DatasetDict containing an IterableDataset."
                    )

            batch_samplers = [
                self.get_batch_sampler(
                    dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns,
                    generator=generator,
                )
                for dataset in train_dataset.values()
            ]

            train_dataset = ConcatDataset(train_dataset.values())
            dataloader_params["batch_sampler"] = self.get_multi_dataset_batch_sampler(
                dataset=train_dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )
        elif isinstance(train_dataset, Dataset):
            dataloader_params["batch_sampler"] = self.get_batch_sampler(
                train_dataset,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                valid_label_columns=data_collator.valid_label_columns,
                generator=generator,
            )
        else:
            raise ValueError(
                "Unsupported `train_dataset` type. Use a Dataset, DatasetDict, or IterableDataset for training."
            )

        self.accelerator.even_batches = False
        self._train_dataloader = DataLoader(train_dataset, **dataloader_params)
        logger.info(
            "Using unsharded train dataloader: batch_sampler=%s",
            type(getattr(self._train_dataloader, "batch_sampler", None)).__name__,
        )
        return self._train_dataloader


def stable_hash63(value: Any) -> int:
    text = "" if value is None else str(value)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


class PositiveOnlyInBatchCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    """Cached MNRL that samples in-batch negatives only from other rows' positives."""

    def __init__(self, *args, negative_sampling_seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._neg_generator = torch.Generator()
        self._neg_generator.manual_seed(negative_sampling_seed)

    def get_in_batch_negatives(self, anchors: list[str], candidates: list[list[str]]):
        batch_size = len(anchors)
        if batch_size <= 1 or not candidates:
            return

        positives = candidates[0]
        if len(positives) != batch_size:
            raise ValueError(
                "PositiveOnlyInBatchCachedMultipleNegativesRankingLoss expects one positive per anchor: "
                f"anchors={batch_size}, positives={len(positives)}"
            )

        # Hard negatives from other rows can be positives for the current anchor.
        # Keep them as hard negatives only for their own row; sample random
        # in-batch negatives from other rows' positives.
        mask = ~torch.eye(batch_size, dtype=torch.bool)
        valid_negative_count = batch_size - 1
        if self.num_negatives is not None and self.num_negatives < valid_negative_count:
            negative_indices = torch.multinomial(mask.float(), int(self.num_negatives), generator=self._neg_generator)
        else:
            all_indices = torch.arange(batch_size).repeat(batch_size, 1)
            negative_indices = all_indices[mask].reshape(batch_size, -1)
        for negative_indices_row in negative_indices.T:
            yield [positives[negative_idx] for negative_idx in negative_indices_row]


def _global_rank_from_env() -> int:
    for key in ("RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return 0


def _rank_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = _global_rank_from_env()
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    return rank, world_size


def _is_global_rank_zero() -> bool:
    rank, _ = _rank_world_size()
    return rank == 0


def _log_rss(tag: str) -> None:
    """Best-effort resident memory logging for long DDP runs."""
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes; macOS reports bytes.
        if os.uname().sysname == "Linux":
            rss_gb = rss_kb / (1024 * 1024)
        else:
            rss_gb = rss_kb / (1024 * 1024 * 1024)
        logger.info("RSS @ %s: %.2f GB", tag, rss_gb)
    except Exception as exc:
        logger.debug("Could not log RSS for %s: %s", tag, exc)


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _show_progress_this_rank() -> bool:
    """Whether tqdm/live progress bars are allowed.

    Disabled by default because tqdm can block distributed jobs by writing too
    frequently to stdout/stderr. Use ENABLE_TQDM=1 only for short interactive
    debugging.
    """
    if _env_flag("ENABLE_TQDM", "0"):
        return True
    if _env_flag("DISABLE_TQDM", "1"):
        return False
    if _env_flag("SHOW_ALL_TQDM", "0"):
        return True
    for env in ("LOCAL_RANK", "SLURM_LOCALID"):
        value = os.environ.get(env)
        if value is not None:
            try:
                return int(value) == 0
            except ValueError:
                return False
    return _is_global_rank_zero()


class _RankZeroConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _is_global_rank_zero()


class _DropOldestQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that never blocks the training thread on logging backpressure."""

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                pass


def _start_liveness_watchdog(log_dir: Path, rank: int, interval_seconds: float = 30.0) -> None:
    """Touch a per-rank heartbeat file outside the logging stack."""
    if rank in _WATCHDOG_THREADS:
        return

    heartbeat_dir = log_dir / "liveness"
    heartbeat_path = heartbeat_dir / f"rank{rank}.lastping"

    def _run() -> None:
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                tmp_path = heartbeat_path.with_suffix(".tmp")
                tmp_path.write_text(str(time.time()), encoding="utf-8")
                tmp_path.replace(heartbeat_path)
            except Exception:
                pass
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_run, name=f"train-liveness-rank{rank}", daemon=True)
    thread.start()
    _WATCHDOG_THREADS[rank] = thread


def _setup_file_logging_all_ranks() -> None:
    """Write INFO logs to per-rank files and keep console output low-volume.

    Rank 0 was observed blocked in pipe_write during evaluation. The safest
    setup is: file logs for observability; console only WARNING+ from rank 0.
    """
    root = logging.getLogger()
    rank, world_size = _rank_world_size()
    marker = f"_cross_encoder_train_file_log_rank_{rank}"
    if getattr(root, marker, False):
        return

    log_dir = Path(os.environ.get("TRAIN_LOG_DIR", "./logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    class _RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = rank
            return True

    rank_filter = _RankFilter()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - rank=%(rank)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_dir / f"train.rank{rank}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    file_handlers: list[logging.Handler] = [fh]

    if rank == 0:
        agg = logging.FileHandler(log_dir / "train.rank0.aggregate.log", encoding="utf-8")
        agg.setLevel(logging.INFO)
        agg.setFormatter(formatter)
        file_handlers.append(agg)

    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=int(os.environ.get("TRAIN_LOG_QUEUE_SIZE", "10000")))
    qh = _DropOldestQueueHandler(log_queue)
    qh.setLevel(logging.INFO)
    qh.addFilter(rank_filter)
    root.addHandler(qh)
    listener = logging.handlers.QueueListener(log_queue, *file_handlers, respect_handler_level=True)
    listener.start()
    _LOG_LISTENERS.append(listener)

    _start_liveness_watchdog(log_dir, rank)

    console_level_name = os.environ.get("CONSOLE_LOG_LEVEL", "WARNING").upper()
    console_level = getattr(logging, console_level_name, logging.WARNING)
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(console_level)
            if not any(isinstance(f, _RankZeroConsoleFilter) for f in handler.filters):
                handler.addFilter(_RankZeroConsoleFilter())
            # Ensure formatter can render %(rank)s if root basicConfig uses it later.
            if not any(f.__class__.__name__ == "_RankFilter" for f in handler.filters):
                handler.addFilter(rank_filter)

    # Also silence HF datasets progress bars where supported.
    if not _show_progress_this_rank():
        try:
            from datasets.utils.logging import disable_progress_bar
            disable_progress_bar()
        except Exception:
            pass

    setattr(root, marker, True)
    logger.info(
        "Per-rank file logging enabled rank=%d/%d file=%s console_level=%s tqdm_enabled=%s",
        rank,
        world_size,
        log_dir / f"train.rank{rank}.log",
        logging.getLevelName(console_level),
        _show_progress_this_rank(),
    )


def _to_clearml_builtin(value: Any) -> Any:
    """Convert config values to ClearML-friendly builtin types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_clearml_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_clearml_builtin(v) for k, v in value.items()}
    return str(value)


def _training_args_for_clearml(training_args: Any) -> dict[str, Any]:
    """Return a safe dictionary for ClearML Task.connect.

    ClearML prints warnings/errors when connected objects contain non-builtin
    fields such as AcceleratorConfig or custom batch_sampler factories. This
    helper keeps scalar/list/dict values and stringifies everything else.
    """
    if hasattr(training_args, "to_dict"):
        raw = training_args.to_dict()
    elif hasattr(training_args, "__dict__"):
        raw = dict(training_args.__dict__)
    else:
        raw = {}

    return {str(k): _to_clearml_builtin(v) for k, v in raw.items()}


def _trainer_metric_name(metric_name: str) -> str:
    return metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"


def _evaluator_metric_name(metric_name: str) -> str:
    return metric_name.removeprefix("eval_")


def _checkpoint_step(path: Path) -> int:
    suffix = path.name.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def _is_full_trainer_checkpoint(path: Path) -> bool:
    return (path / "trainer_state.json").is_file()


def _find_latest_full_checkpoint(output_dir: str | Path) -> Path | None:
    output_dir = Path(output_dir)
    full = [
        p
        for p in output_dir.glob("checkpoint-*")
        if p.is_dir() and _is_full_trainer_checkpoint(p)
    ]
    if not full:
        return None
    return max(full, key=_checkpoint_step)


def _resolve_resume_checkpoint(output_dir: str | Path) -> str | bool:
    """Resume path from RESUME_FROM_CHECKPOINT env var.

    - unset / false / 0: train from scratch
    - latest / auto: newest checkpoint with trainer_state.json
    - otherwise: explicit checkpoint directory
    """
    explicit = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()
    if not explicit or explicit.lower() in {"0", "false", "no", "none"}:
        return False
    output_dir = Path(output_dir)
    if explicit.lower() in {"latest", "auto"}:
        latest = _find_latest_full_checkpoint(output_dir)
        if latest is None:
            raise FileNotFoundError(f"No full checkpoints (trainer_state.json) in {output_dir}")
        return str(latest)
    path = Path(explicit)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    if not _is_full_trainer_checkpoint(path):
        raise ValueError(
            f"{path} has no trainer_state.json (model-only export). "
            f"Use a full checkpoint or RESUME_FROM_CHECKPOINT=latest."
        )
    return str(path)


def _find_sampler_with_schedule_len(obj: Any, seen: set[int] | None = None) -> Any | None:
    if obj is None:
        return None
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return None
    seen.add(obj_id)
    if hasattr(obj, "_last_schedule_len"):
        return obj
    for attr in ("batch_sampler", "sampler"):
        found = _find_sampler_with_schedule_len(getattr(obj, attr, None), seen)
        if found is not None:
            return found
    return None


class TrainingPhaseTimingCallback(TrainerCallback):
    def __init__(self, slow_step_seconds: float = 120.0, heartbeat_every: int = 50) -> None:
        self.slow_step_seconds = slow_step_seconds
        self.heartbeat_every = heartbeat_every
        self._step_t0: float | None = None
        self._last_step_end: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        if _is_global_rank_zero():
            logger.info("Callback: training started (global_step=%s)", state.global_step)
            logger.info(
                "Trainer state: max_steps=%s, num_train_epochs=%s, args.num_train_epochs=%s",
                state.max_steps,
                state.num_train_epochs,
                args.num_train_epochs,
            )
            _log_rss("train_begin")

    def on_epoch_begin(self, args, state, control, **kwargs):
        if _is_global_rank_zero():
            logger.info("Callback: epoch begin (epoch=%.4f, global_step=%s)", state.epoch, state.global_step)
            _log_rss(f"epoch_begin_{state.epoch}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if _is_global_rank_zero():
            logger.info("Callback: epoch end (epoch=%.4f, global_step=%s)", state.epoch, state.global_step)
            _log_rss(f"epoch_end_{state.epoch}")

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_t0 is None:
            return
        dt = time.perf_counter() - self._step_t0
        self._last_step_end = time.perf_counter()
        if not _is_global_rank_zero():
            return
        if dt >= self.slow_step_seconds:
            logger.warning(
                "Callback: slow training step global_step=%s took %.1fs (threshold=%.0fs)",
                state.global_step,
                dt,
                self.slow_step_seconds,
            )
            _log_rss(f"slow_step_{state.global_step}")
        gs = int(state.global_step)
        if self.heartbeat_every > 0 and gs > 0 and gs % self.heartbeat_every == 0:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e9
                logger.info("Callback: heartbeat global_step=%s cuda_mem_allocated≈%.2f GB", gs, alloc)
            _log_rss(f"heartbeat_step_{gs}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not _is_global_rank_zero():
            return
        metrics = metrics if metrics is not None else kwargs.get("metrics")
        metric_name = getattr(args, "metric_for_best_model", None)
        metric_value = (metrics or {}).get(metric_name) if metric_name else None
        logger.info(
            "Callback: evaluation finished; metric_for_best_model=%s value=%s best_metric=%s keys=%s",
            metric_name,
            metric_value,
            getattr(state, "best_metric", None),
            list(metrics or {}),
        )
        _log_rss("evaluation_finished")


class LogitRangeCallback(TrainerCallback):
    """Periodically log raw reranker score range on a fixed eval mini-batch."""

    def __init__(
        self,
        model: CrossEncoder,
        eval_samples: list[dict],
        *,
        every_steps: int = 100,
        until_step: int = 1000,
        max_pairs: int = 64,
    ) -> None:
        self.model = model
        self.every_steps = every_steps
        self.until_step = until_step
        self.pairs: list[list[str]] = []
        for sample in eval_samples:
            query = sample.get("query")
            docs = list(sample.get("positive") or []) + list(sample.get("negative") or [])
            for doc in docs:
                if query and doc:
                    self.pairs.append([str(query), str(doc)])
                if len(self.pairs) >= max_pairs:
                    break
            if len(self.pairs) >= max_pairs:
                break

    def on_step_end(self, args, state, control, **kwargs):
        if not _is_global_rank_zero() or not self.pairs:
            return
        step = int(state.global_step)
        if step <= 0 or step > self.until_step or step % self.every_steps != 0:
            return
        original_module = self.model.model
        was_training = bool(getattr(original_module, "training", False))
        if isinstance(self.model.model, torch.nn.parallel.DistributedDataParallel):
            self.model.model = self.model.model.module
        try:
            scores = self.model.predict(
                self.pairs,
                batch_size=min(32, len(self.pairs)),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        finally:
            self.model.model = original_module
            self.model.model.train(was_training)
        scores = np.asarray(scores, dtype=np.float32)
        logger.info(
            "Logit range probe step=%d pairs=%d min=%.4f mean=%.4f max=%.4f",
            step,
            len(self.pairs),
            float(scores.min()),
            float(scores.mean()),
            float(scores.max()),
        )


class Evaluator(CrossEncoderRerankingEvaluator):
    """Streaming distributed evaluator: no all_pairs materialization."""
    LANG_BUCKETS = ("ru", "en", "ru-en", "en-ru")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_headers = ["epoch", "steps", "MAP", f"MRR@{self.at_k}", "Accuracy@1", "Accuracy@3", "Accuracy@5"]
        self.primary_metric = "map"

    def compute_metrics(self, y_true, y_pred):
        y_true_arr = np.asarray(y_true, dtype=np.int8)
        y_pred_arr = np.asarray(y_pred, dtype=np.float32)
        ranking = np.argsort(y_pred_arr)[::-1]
        top_relevance = y_true_arr[ranking[: self.at_k]]
        mrr = 0.0
        hits = np.flatnonzero(top_relevance)
        if hits.size:
            mrr = 1.0 / (int(hits[0]) + 1)
        ap = average_precision_score(y_true_arr, y_pred_arr) if y_true_arr.any() else 0.0
        return {
            "mrr": float(mrr),
            "ap": float(ap),
            "accuracy@1": int(y_true_arr[ranking[:1]].any()),
            "accuracy@3": int(y_true_arr[ranking[:3]].any()),
            "accuracy@5": int(y_true_arr[ranking[:5]].any()),
        }

    @staticmethod
    def _lang_key(value: Any) -> str | None:
        text = _first_non_empty(value).lower().strip()
        if not text:
            return None
        normalized = re.sub(r"[^a-zа-я]+", "-", text).strip("-")
        tokens = [token for token in normalized.split("-") if token]

        if normalized in {"ru", "rus", "russian", "рус", "русский"}:
            return "ru"
        if normalized in {"en", "eng", "english", "англ", "английский"}:
            return "en"

        canonical_tokens: list[str] = []
        for token in tokens:
            if token in {"ru", "rus", "russian", "рус", "русский"}:
                canonical_tokens.append("ru")
            elif token in {"en", "eng", "english", "англ", "английский"}:
                canonical_tokens.append("en")

        if len(canonical_tokens) >= 2:
            first, second = canonical_tokens[0], canonical_tokens[1]
            if first == "ru" and second == "en":
                return "ru-en"
            if first == "en" and second == "ru":
                return "en-ru"
            return first

        if "ru" in canonical_tokens:
            return "ru"
        if "en" in canonical_tokens:
            return "en"
        return None

    @staticmethod
    def _empty_metric_sums() -> dict[str, float]:
        return {"ap": 0.0, "mrr": 0.0, "accuracy@1": 0.0, "accuracy@3": 0.0, "accuracy@5": 0.0, "count": 0.0}

    @staticmethod
    def _add_metric_sums(target: dict[str, float], metrics: dict[str, float]) -> None:
        target["ap"] += float(metrics["ap"])
        target["mrr"] += float(metrics["mrr"])
        target["accuracy@1"] += float(metrics["accuracy@1"])
        target["accuracy@3"] += float(metrics["accuracy@3"])
        target["accuracy@5"] += float(metrics["accuracy@5"])
        target["count"] += 1.0

    @staticmethod
    def _mean_metric_sums(prefix: str, sums: dict[str, float], at_k: int) -> dict[str, float]:
        count = sums["count"]
        if count <= 0:
            return {}
        return {
            f"map_{prefix}": sums["ap"] / count,
            f"mrr@{at_k}_{prefix}": sums["mrr"] / count,
            f"accuracy@1_{prefix}": sums["accuracy@1"] / count,
            f"accuracy@3_{prefix}": sums["accuracy@3"] / count,
            f"accuracy@5_{prefix}": sums["accuracy@5"] / count,
        }

    @staticmethod
    def _is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    @staticmethod
    def _is_main_process() -> bool:
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        is_main = self._is_main_process()
        rank, world_size = _rank_world_size()

        if is_main:
            logger.info("Starting streaming evaluation on %d rank(s)", world_size)

        total_queries = 0
        sum_ap = 0.0
        sum_mrr = 0.0
        sum_acc1 = 0.0
        sum_acc3 = 0.0
        sum_acc5 = 0.0
        lang_sums = {lang: self._empty_metric_sums() for lang in self.LANG_BUCKETS}
        num_positives: list[int] = []
        num_negatives: list[int] = []

        original_module = model.model
        if isinstance(model.model, torch.nn.parallel.DistributedDataParallel):
            model.model = model.model.module

        # Do not use tqdm by default here: frequent stderr writes can block the
        # distributed job when stdout/stderr pipes fill. File logs below provide
        # observability without pipe pressure.
        progress = None
        iterator = self.samples
        if self.show_progress_bar and _show_progress_this_rank():
            progress = tqdm(
                self.samples,
                desc=f"Eval rank {rank}: queries",
                unit="query",
                leave=False,
            )
            iterator = progress

        eval_log_every = int(os.environ.get("EVAL_LOG_EVERY", "1000"))
        eval_state_every = int(os.environ.get("EVAL_STATE_EVERY", "100"))
        slow_eval_seconds = float(os.environ.get("SLOW_EVAL_SAMPLE_SECONDS", "60"))
        log_dir = Path(os.environ.get("TRAIN_LOG_DIR", "./logs")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        state_path = log_dir / f"eval_state.rank{rank}.json"
        last_eval_log_t = time.perf_counter()

        try:
            for idx, instance in enumerate(iterator):
                query = instance["query"]
                positive = instance["positive"]
                if isinstance(positive, str):
                    positive = [positive]
                negative = instance.get("negative", None)
                documents = instance.get("documents", None)

                if documents:
                    if self.always_rerank_positives:
                        docs = positive + [doc for doc in documents if doc not in positive]
                        is_relevant = [1] * len(positive) + [0] * (len(docs) - len(positive))
                    else:
                        docs = documents
                        is_relevant = [int(sample in positive) for sample in documents]
                else:
                    negative = negative or []
                    docs = positive + negative
                    is_relevant = [1] * len(positive) + [0] * len(negative)

                total_queries += 1
                num_positives.append(len(positive))
                num_negatives.append(len(is_relevant) - sum(is_relevant))

                if not any(is_relevant):
                    continue

                sample_t0 = time.perf_counter()
                num_pos = len(positive)
                num_neg = len(is_relevant) - sum(is_relevant)
                num_docs = len(docs)
                query_chars = len(str(query))
                max_doc_chars = max((len(str(doc)) for doc in docs), default=0)
                total_doc_chars = sum(len(str(doc)) for doc in docs)

                if eval_state_every > 0 and idx % eval_state_every == 0:
                    try:
                        tmp_state = state_path.with_suffix(".tmp")
                        tmp_state.write_text(
                            json.dumps(
                                {
                                    "rank": rank,
                                    "idx": idx,
                                    "total": len(self.samples),
                                    "num_docs": num_docs,
                                    "positives": num_pos,
                                    "negatives": num_neg,
                                    "query_chars": query_chars,
                                    "max_doc_chars": max_doc_chars,
                                    "total_doc_chars": total_doc_chars,
                                    "time": time.time(),
                                },
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
                        )
                        tmp_state.replace(state_path)
                    except Exception as exc:
                        logger.debug("Could not write eval state rank=%d idx=%d: %s", rank, idx, exc)

                if num_docs > 128 or max_doc_chars > 50_000 or total_doc_chars > 500_000:
                    logger.warning(
                        "Large eval sample BEFORE predict: rank=%d idx=%d/%d docs=%d pos=%d neg=%d "
                        "query_chars=%d max_doc_chars=%d total_doc_chars=%d",
                        rank,
                        idx,
                        len(self.samples),
                        num_docs,
                        num_pos,
                        num_neg,
                        query_chars,
                        max_doc_chars,
                        total_doc_chars,
                    )

                pairs = [[query, doc] for doc in docs]
                scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                sample_dt = time.perf_counter() - sample_t0
                if sample_dt >= slow_eval_seconds:
                    logger.warning(
                        "Slow eval sample AFTER predict: rank=%d idx=%d/%d took=%.1fs docs=%d pos=%d neg=%d "
                        "query_chars=%d max_doc_chars=%d total_doc_chars=%d",
                        rank,
                        idx,
                        len(self.samples),
                        sample_dt,
                        num_docs,
                        num_pos,
                        num_neg,
                        query_chars,
                        max_doc_chars,
                        total_doc_chars,
                    )

                metrics = self.compute_metrics(is_relevant, scores)
                sum_ap += metrics["ap"]
                sum_mrr += metrics["mrr"]
                sum_acc1 += metrics["accuracy@1"]
                sum_acc3 += metrics["accuracy@3"]
                sum_acc5 += metrics["accuracy@5"]
                lang_key = self._lang_key(instance.get("lang"))
                if lang_key in lang_sums:
                    self._add_metric_sums(lang_sums[lang_key], metrics)

                now = time.perf_counter()
                if idx > 0 and (
                    (eval_log_every > 0 and idx % eval_log_every == 0)
                    or (now - last_eval_log_t >= 60)
                ):
                    logger.info(
                        "Eval progress rank=%d processed=%d/%d docs=%d pos=%d neg=%d last_sample=%.2fs",
                        rank,
                        idx,
                        len(self.samples),
                        num_docs,
                        num_pos,
                        num_neg,
                        sample_dt,
                    )
                    last_eval_log_t = now
        finally:
            if progress is not None:
                progress.close()
            model.model = original_module

        local_count = float(total_queries)
        logger.info(
            "Eval rank=%d finished local evaluation: queries=%d sum_ap=%.4f; entering metric all_reduce=%s",
            rank,
            total_queries,
            sum_ap,
            self._is_distributed(),
        )
        if self._is_distributed():
            device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
            values = torch.tensor(
                [sum_ap, sum_mrr, sum_acc1, sum_acc3, sum_acc5, local_count],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            logger.info("Eval rank=%d finished metric all_reduce", rank)
            sum_ap, sum_mrr, sum_acc1, sum_acc3, sum_acc5, global_count = values.cpu().tolist()

            lang_values = torch.tensor(
                [
                    value
                    for lang in self.LANG_BUCKETS
                    for value in (
                        lang_sums[lang]["ap"],
                        lang_sums[lang]["mrr"],
                        lang_sums[lang]["accuracy@1"],
                        lang_sums[lang]["accuracy@3"],
                        lang_sums[lang]["accuracy@5"],
                        lang_sums[lang]["count"],
                    )
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(lang_values, op=dist.ReduceOp.SUM)
            lang_numbers = lang_values.cpu().tolist()
            for lang_idx, lang in enumerate(self.LANG_BUCKETS):
                offset = lang_idx * 6
                lang_sums[lang] = {
                    "ap": lang_numbers[offset],
                    "mrr": lang_numbers[offset + 1],
                    "accuracy@1": lang_numbers[offset + 2],
                    "accuracy@3": lang_numbers[offset + 3],
                    "accuracy@5": lang_numbers[offset + 4],
                    "count": lang_numbers[offset + 5],
                }
        else:
            global_count = local_count

        denom = global_count if global_count > 0 else 1.0
        mean_ap = sum_ap / denom
        mean_mrr = sum_mrr / denom
        mean_accuracy_at_1 = sum_acc1 / denom
        mean_accuracy_at_3 = sum_acc3 / denom
        mean_accuracy_at_5 = sum_acc5 / denom

        metrics = {
            "map": mean_ap,
            f"mrr@{self.at_k}": mean_mrr,
            "accuracy@1": mean_accuracy_at_1,
            "accuracy@3": mean_accuracy_at_3,
            "accuracy@5": mean_accuracy_at_5,
        }
        for lang in self.LANG_BUCKETS:
            metrics.update(self._mean_metric_sums(lang, lang_sums[lang], self.at_k))

        if is_main:
            if num_positives:
                logger.info(
                    "Local rank0 eval stats: queries=%d positives mean=%.2f negatives mean=%.2f",
                    total_queries,
                    float(np.mean(num_positives)),
                    float(np.mean(num_negatives)),
                )
            logger.info("Global MAP: %.4f", mean_ap)
            logger.info("Global MRR@%s: %.4f", self.at_k, mean_mrr)
            logger.info("Global Accuracy@1/3/5: %.4f / %.4f / %.4f", mean_accuracy_at_1, mean_accuracy_at_3, mean_accuracy_at_5)
            for lang in self.LANG_BUCKETS:
                if lang_sums[lang]["count"] > 0:
                    logger.info(
                        "Global %s MAP: %.4f over %.0f query group(s)",
                        lang,
                        lang_sums[lang]["ap"] / lang_sums[lang]["count"],
                        lang_sums[lang]["count"],
                    )

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        if is_main and output_path is not None and self.write_csv:
            import csv
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / self.csv_file
            exists = csv_path.is_file()
            with csv_path.open("a" if exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps, mean_ap, mean_mrr, mean_accuracy_at_1, mean_accuracy_at_3, mean_accuracy_at_5])

        return metrics


class RerankingSample(BaseModel):
    query: str = Field(..., min_length=1)
    positive: list[str] = Field(..., min_length=1)
    negative: list[str] = Field(default_factory=list)
    lang: str | None = None

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: Any) -> str:
        if isinstance(value, list):
            return str(value[0]) if value and value[0] else ""
        return str(value) if value else ""

    @field_validator("positive", mode="before")
    @classmethod
    def normalize_positives(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if item and str(item).strip()]
        return []

    @field_validator("negative", mode="before")
    @classmethod
    def normalize_negatives(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if item and str(item).strip()]
        return []

    @field_validator("lang", mode="before")
    @classmethod
    def normalize_lang(cls, value: Any) -> str | None:
        text = _first_non_empty(value)
        return text or None

    @model_validator(mode="after")
    def ensure_min_content(self) -> "RerankingSample":
        if not self.query.strip():
            raise ValueError("Empty query after normalization")
        if not self.positive:
            raise ValueError("No valid positives after normalization")
        return self


def _first_non_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        for item in value:
            text = _first_non_empty(item)
            if text:
                return text
        return ""
    return str(value).strip()


def _batch_len(batch: dict[str, list[Any]]) -> int:
    if not batch:
        return 0
    return len(next(iter(batch.values())))



_NEGATIVE_COL_RE = re.compile(r"^negative_?(\d+)$")


def _negative_col_sort_key(column_name: str) -> tuple[int, str]:
    match = _NEGATIVE_COL_RE.match(column_name)
    if match:
        return int(match.group(1)), column_name
    return 10**9, column_name


def get_training_text_columns(column_names: list[str], num_hard_negatives: int = 0) -> list[str]:
    """Return only text columns that should be consumed by the CrossEncoder loss.

    Bucket/helper columns such as query_id, query_key64, positive_key64,
    bucket_id, dataset, lang, etc. must not reach CrossEncoderTrainer. If they
    stay in the Dataset, SentenceTransformers can treat a numeric helper column
    as a text field and pass an int into the tokenizer, producing:
    "TypeError: TextInputSequence must be str".

    num_hard_negatives controls how many hard-negative columns are kept.
    Raises ValueError if the dataset has fewer negative columns than requested.
    Pass 0 to train without any hard negatives.
    """
    columns = set(column_names)
    keep: list[str] = []

    if "query" in columns:
        keep.append("query")
    elif "anchor" in columns:
        keep.append("anchor")
    else:
        raise ValueError(
            "Training dataset must contain either 'query' or 'anchor'. "
            f"Available columns: {column_names}"
        )

    if "positive" not in columns:
        raise ValueError(
            "Training dataset must contain 'positive'. "
            f"Available columns: {column_names}"
        )
    keep.append("positive")

    negative_columns = sorted(
        [name for name in column_names if _NEGATIVE_COL_RE.match(name)],
        key=_negative_col_sort_key,
    )

    if num_hard_negatives > len(negative_columns):
        raise ValueError(
            f"num_hard_negatives={num_hard_negatives} requested but dataset only contains "
            f"{len(negative_columns)} negative column(s): {negative_columns}"
        )

    keep.extend(negative_columns[:num_hard_negatives])
    logger.info(
        "Hard negatives: using %d of %d available negative column(s).",
        num_hard_negatives,
        len(negative_columns),
    )

    return keep


def sanitize_training_dataset_for_loss(
    dataset,
    *,
    num_hard_negatives: int = 0,
    batch_size: int = 50_000,
):
    """Remove helper columns and force remaining training text columns to str.

    This must run after build_key_maps(...), because build_key_maps needs
    query_key64/positive_key64 if present. The operation preserves row order and
    row count, so sampler groups and index_to_* maps remain valid.
    """
    original_columns = list(dataset.column_names)
    text_columns = get_training_text_columns(original_columns, num_hard_negatives=num_hard_negatives)
    remove_columns = [name for name in original_columns if name not in text_columns]

    logger.info(
        "Sanitizing training dataset for CrossEncoder loss. Keeping text columns=%s; removing columns=%s",
        text_columns,
        remove_columns,
    )

    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    def normalize_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for column in text_columns:
            normalized[column] = [_first_non_empty(value) for value in batch[column]]
        return normalized

    dataset = dataset.map(
        normalize_batch,
        batched=True,
        batch_size=batch_size,
        desc="Normalize training text columns",
    )

    # Validate a small prefix and fail before the first expensive training step.
    sample_n = min(2048, len(dataset))
    if sample_n > 0:
        sample = dataset.select(range(sample_n))
        for row_idx, row in enumerate(sample):
            for column in text_columns:
                value = row[column]
                if not isinstance(value, str):
                    raise TypeError(
                        f"Column {column!r} still has non-string value after normalization "
                        f"at sampled row {row_idx}: {type(value).__name__}"
                    )
                if column in ("query", "anchor", "positive") and not value:
                    raise ValueError(
                        f"Required column {column!r} is empty after normalization "
                        f"at sampled row {row_idx}."
                    )

    logger.info(
        "Training dataset sanitized. Final columns=%s rows=%d",
        list(dataset.column_names),
        len(dataset),
    )
    return dataset


def prepare_reranking_samples(
    dataset,
    num_negatives: int,
    max_samples: int | None = None,
    query_field: str = "query",
    batch_size: int = 50_000,
) -> list[dict]:
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    family_positive_texts: dict[int, dict[int, str]] = defaultdict(dict)
    parse_errors = 0
    total_rows = len(dataset)
    columns = set(dataset.column_names)

    iterator = dataset.iter(batch_size=batch_size)
    progress = tqdm(
        total=total_rows,
        desc="Prepare eval samples",
        unit="row",
        leave=True,
        disable=not _show_progress_this_rank(),
    )

    processed_rows = 0
    last_prepare_log_t = time.perf_counter()
    try:
        for batch in iterator:
            n = _batch_len(batch)
            processed_rows += n
            progress.update(n)
            now = time.perf_counter()
            if now - last_prepare_log_t >= 30:
                logger.info(
                    "Prepare eval samples progress: rows=%d/%d groups=%d family_keys=%d parse_errors=%d",
                    processed_rows,
                    total_rows,
                    len(grouped),
                    len(family_positive_texts),
                    parse_errors,
                )
                last_prepare_log_t = now
            for i in range(n):
                try:
                    query_raw = batch.get(query_field, batch.get("anchor", batch.get("query")))[i]
                    query = _first_non_empty(query_raw)
                    if not query:
                        raise ValueError("Empty query")

                    if "query_key64" in columns:
                        family_key = int(batch["query_key64"][i])
                    elif "query_id" in columns:
                        family_key = stable_hash63(_first_non_empty(batch["query_id"][i]) or query)
                    else:
                        family_key = stable_hash63(query)
                    query_key = stable_hash63(query)

                    pos_raw = batch.get("positive", [[]])[i]
                    positive_texts: dict[int, str] = {}
                    if isinstance(pos_raw, list):
                        for p in pos_raw:
                            text = _first_non_empty(p)
                            if text:
                                positive_texts[stable_hash63(text)] = text
                    else:
                        pos = _first_non_empty(pos_raw)
                        if pos:
                            if "positive_key64" in columns:
                                positive_texts[int(batch["positive_key64"][i])] = pos
                            else:
                                positive_texts[stable_hash63(pos)] = pos
                    if not positive_texts:
                        raise ValueError("No valid positives")

                    negative_texts: dict[int, str] = {}
                    for j in range(1, num_negatives + 1):
                        for col in (f"negative{j}", f"negative_{j}"):
                            if col in batch:
                                neg = _first_non_empty(batch[col][i])
                                if neg:
                                    key_col_candidates = (
                                        f"negative_key64_{j}",
                                        f"negative{j}_key64",
                                        f"negative_{j}_key64",
                                    )
                                    neg_key = None
                                    for key_col in key_col_candidates:
                                        if key_col in columns:
                                            neg_key = int(batch[key_col][i])
                                            break
                                    if neg_key is None:
                                        neg_key = stable_hash63(neg)
                                    negative_texts[neg_key] = neg
                                break

                    family_positive_texts[family_key].update(positive_texts)
                    key = (family_key, query_key)
                    if key not in grouped:
                        grouped[key] = {
                            "query": query,
                            "family_key": family_key,
                            "positive_texts": {},
                            "negative_texts": {},
                            "lang": _first_non_empty(batch["lang"][i]) if "lang" in columns else None,
                        }
                    grouped[key]["positive_texts"].update(positive_texts)
                    grouped[key]["negative_texts"].update(negative_texts)
                    if not grouped[key].get("lang") and "lang" in columns:
                        grouped[key]["lang"] = _first_non_empty(batch["lang"][i]) or None
                except Exception as exc:
                    parse_errors += 1
                    if parse_errors <= 5:
                        logger.debug("Eval row skipped: %s", exc)
    finally:
        progress.close()

    valid_samples: list[dict] = []
    invalid_count = 0

    group_iter = tqdm(
        grouped.values(),
        desc="Validate eval query groups",
        unit="query",
        leave=True,
        disable=not _show_progress_this_rank(),
    )

    last_validate_log_t = time.perf_counter()
    for group_idx, data in enumerate(group_iter):
        now = time.perf_counter()
        if now - last_validate_log_t >= 30:
            logger.info(
                "Validate eval groups progress: groups=%d/%d valid=%d invalid=%d",
                group_idx,
                len(grouped),
                len(valid_samples),
                invalid_count,
            )
            last_validate_log_t = now
        try:
            family_key = data["family_key"]
            positive_texts = dict(family_positive_texts[family_key])
            positive_texts.update(data["positive_texts"])
            negative_texts = {
                key: text
                for key, text in data["negative_texts"].items()
                if key not in positive_texts
            }
            sample = RerankingSample(
                query=data["query"],
                positive=list(positive_texts.values()),
                negative=list(negative_texts.values()),
                lang=data.get("lang"),
            )
            valid_samples.append(sample.model_dump())
        except (ValidationError, ValueError) as exc:
            invalid_count += 1
            if invalid_count <= 5:
                logger.debug("Eval group skipped: %s", exc)

    if parse_errors or invalid_count:
        logger.warning(
            "prepare_reranking_samples skipped %d rows and %d groups out of %d rows",
            parse_errors,
            invalid_count,
            total_rows,
        )

    if valid_samples:
        avg_pos = sum(len(s["positive"]) for s in valid_samples) / len(valid_samples)
        avg_neg = sum(len(s["negative"]) for s in valid_samples) / len(valid_samples)
    else:
        avg_pos = 0.0
        avg_neg = 0.0

    logger.info(
        "Prepared %d valid evaluation samples from %d local rows (avg %.1f positives, %.1f negatives)",
        len(valid_samples),
        total_rows,
        avg_pos,
        avg_neg,
    )
    return valid_samples


def load_plain_parquets_with_groups(data_dir: str) -> tuple[Any, dict[str, list[int]]]:
    """Fallback non-bucketed loader. Prefer bucketed datasets for DDP memory safety."""
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise ValueError(f"No .parquet files found in {data_dir}")

    data_files: list[str] = []
    row_counts: list[tuple[str, int]] = []
    for path in tqdm(files, desc=f"Metadata {data_dir}", unit="file", disable=not _show_progress_this_rank()):
        rows = pq.read_metadata(str(path)).num_rows
        logger.info("Metadata %s: %d rows", path.name, rows)
        if rows > 0:
            data_files.append(str(path))
            row_counts.append((path.stem, rows))
        else:
            logger.warning("Skipping empty parquet file: %s", path)

    dataset = load_dataset("parquet", data_files={"train": data_files}, split="train", keep_in_memory=False)

    groups: dict[str, list[int]] = {}
    offset = 0
    for name, rows in row_counts:
        groups[name] = list(range(offset, offset + rows))
        offset += rows

    if len(dataset) != offset:
        raise RuntimeError(f"Loaded dataset length mismatch: expected {offset}, got {len(dataset)}")
    return dataset, groups


def _resolve_bucket_root_and_split(path: str | Path, default_split: str) -> tuple[Path, str]:
    path = Path(path).resolve()
    if (path / "manifest.json").exists():
        return path, default_split
    if (path.parent / "manifest.json").exists():
        return path.parent, path.name
    raise FileNotFoundError(
        f"Could not find manifest.json at {path}/manifest.json or {path.parent}/manifest.json. "
        "Run prepare_buckets.py first or pass a non-bucketed parquet directory."
    )


def _assign_entries_modulo(entries: list[dict[str, Any]], rank: int, world_size: int) -> list[dict[str, Any]]:
    return [e for e in entries if int(e["bucket_id"]) % world_size == rank]


def _assign_entries_greedy_per_dataset(entries: list[dict[str, Any]], rank: int, world_size: int) -> list[dict[str, Any]]:
    # Assign whole (dataset,bucket_id) groups greedily by total rows, separately per dataset.
    dataset_bucket_rows: dict[tuple[str, int], int] = defaultdict(int)
    grouped_entries: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = (str(entry["dataset"]), int(entry["bucket_id"]))
        dataset_bucket_rows[key] += int(entry["rows"])
        grouped_entries[key].append(entry)

    selected_keys: set[tuple[str, int]] = set()
    by_dataset: dict[str, list[tuple[tuple[str, int], int]]] = defaultdict(list)
    for key, rows in dataset_bucket_rows.items():
        by_dataset[key[0]].append((key, rows))

    for dataset_name, bucket_items in by_dataset.items():
        loads = [0 for _ in range(world_size)]
        for key, rows in sorted(bucket_items, key=lambda x: x[1], reverse=True):
            target = min(range(world_size), key=lambda r: loads[r])
            loads[target] += rows
            if target == rank:
                selected_keys.add(key)

    selected: list[dict[str, Any]] = []
    for key in selected_keys:
        selected.extend(grouped_entries[key])
    return selected


def load_bucketed_split(
    path: str | Path,
    split: str,
    rank: int,
    world_size: int,
    assignment: str = "greedy",
) -> tuple[Any, dict[str, list[int]]]:
    root, resolved_split = _resolve_bucket_root_and_split(path, split)
    manifest_path = root / "manifest.json"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest["splits"].get(resolved_split, [])
    if not entries:
        logger.warning("No entries for split=%s in manifest=%s", resolved_split, manifest_path)
        return Dataset.from_dict({}), {}

    if assignment == "greedy":
        selected = _assign_entries_greedy_per_dataset(entries, rank, world_size)
    elif assignment == "modulo":
        selected = _assign_entries_modulo(entries, rank, world_size)
    else:
        raise ValueError(f"Unknown bucket assignment strategy: {assignment}")

    selected = sorted(selected, key=lambda e: (str(e["dataset"]), int(e["bucket_id"]), str(e["path"])))
    logger.info(
        "Bucketed load split=%s rank=%d/%d assignment=%s selected_files=%d/%d selected_rows=%d",
        resolved_split,
        rank,
        world_size,
        assignment,
        len(selected),
        len(entries),
        sum(int(e["rows"]) for e in selected),
    )

    if not selected:
        logger.warning("Rank %d selected no files for split=%s", rank, resolved_split)
        return Dataset.from_dict({}), {}

    data_files = [str(root / entry["path"]) for entry in selected]
    groups: dict[str, list[int]] = defaultdict(list)
    offset = 0
    for entry in selected:
        rows = int(entry["rows"])
        dataset_name = str(entry["dataset"])
        groups[dataset_name].extend(range(offset, offset + rows))
        offset += rows

    for entry in tqdm(selected, desc=f"Rank {rank} metadata {resolved_split}", unit="file", disable=not _show_progress_this_rank()):
        file_path = root / entry["path"]
        actual_rows = pq.read_metadata(str(file_path)).num_rows
        if actual_rows != int(entry["rows"]):
            raise RuntimeError(f"Manifest row mismatch for {file_path}: manifest={entry['rows']} actual={actual_rows}")

    logger.info("Rank %d loading %d parquet file(s) for split=%s", rank, len(data_files), resolved_split)
    dataset = load_dataset("parquet", data_files={"train": data_files}, split="train", keep_in_memory=False)
    if len(dataset) != offset:
        raise RuntimeError(f"Bucketed dataset length mismatch for rank={rank}: expected={offset}, got={len(dataset)}")

    logger.info("Rank %d loaded split=%s rows=%d groups=%s", rank, resolved_split, len(dataset), {k: len(v) for k, v in groups.items()})
    return dataset, dict(groups)


def build_key_maps(dataset, no_duplicates: bool, batch_size: int = 100_000) -> tuple[dict[int, int], dict[int, int]]:
    if not no_duplicates:
        return {}, {}

    index_to_anchor: dict[int, int] = {}
    index_to_positive_key: dict[int, int] = {}
    columns = set(dataset.column_names)
    offset = 0

    progress = tqdm(
        total=len(dataset),
        desc="Build sampler key maps",
        unit="row",
        leave=True,
        disable=not _show_progress_this_rank(),
    )

    last_keymap_log_t = time.perf_counter()
    try:
        for batch in dataset.iter(batch_size=batch_size):
            n = _batch_len(batch)
            for i in range(n):
                local_idx = offset + i

                if "query_key64" in columns:
                    anchor_key = int(batch["query_key64"][i])
                elif "query_id" in columns:
                    anchor_key = stable_hash63(_first_non_empty(batch["query_id"][i]))
                elif "query" in columns:
                    anchor_key = stable_hash63(_first_non_empty(batch["query"][i]))
                elif "anchor" in columns:
                    anchor_key = stable_hash63(_first_non_empty(batch["anchor"][i]))
                else:
                    anchor_key = -1 - local_idx
                index_to_anchor[local_idx] = int(anchor_key)

                if "positive_key64" in columns:
                    pos_key = int(batch["positive_key64"][i])
                elif "positive" in columns:
                    pos_key = stable_hash63(_first_non_empty(batch["positive"][i]))
                else:
                    pos_key = stable_hash63(f"__missing_positive__:{local_idx}")
                index_to_positive_key[local_idx] = int(pos_key)

            offset += n
            progress.update(n)
            now = time.perf_counter()
            if now - last_keymap_log_t >= 30:
                logger.info("Build sampler key maps progress: rows=%d/%d", offset, len(dataset))
                last_keymap_log_t = now
    finally:
        progress.close()

    return index_to_anchor, index_to_positive_key


def train_cross_encoder_reranker(
    model_name_or_path: str,
    train_dataset_path: str,
    eval_dataset_path: str,
    training_args,
    mini_batch_size: int = 128,
    num_hard_negatives: int = 8,
    num_negatives: int = 8,
    num_random_negatives: int = 0,
    language: str = "ru+en",
    license: str = "apache-2.0",
    model_card_name: str | None = None,
    use_clearml: bool = True,
    clearml_project_name: str = "CrossEncoders",
    clearml_task_name: str | None = None,
    seed: int = 12,
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 1e-3,
    max_length: int = 4096,
    no_duplicates: bool = False,
    bucket_assignment: str = "greedy",
    loss_scale: float = 10.0,
) -> CrossEncoder:
    _setup_file_logging_all_ranks()
    rank, world_size = _rank_world_size()
    training_args.metric_for_best_model = _trainer_metric_name(training_args.metric_for_best_model)

    task = None
    if use_clearml and _is_global_rank_zero():
        task_name = clearml_task_name or f"reranker-{Path(model_name_or_path).name}-training"
        task = Task.init(project_name=clearml_project_name, task_name=task_name, task_type=Task.TaskTypes.training)
        task.connect(_training_args_for_clearml(training_args), name="training_args")
        task.connect({"model_name_or_path": model_name_or_path, "num_negatives": num_negatives, "seed": seed}, name="hyperparameters")
    elif use_clearml:
        logger.info("ClearML Task.init skipped on rank %s; report_to disabled", rank)
        try:
            training_args.report_to = "none"
        except Exception:
            object.__setattr__(training_args, "report_to", "none")

    logger.info("Loading model: %s", model_name_or_path)
    short_model_name = model_name_or_path if "/" not in model_name_or_path else model_name_or_path.split("/")[-1]
    model = CrossEncoder(
        model_name_or_path,
        max_length=max_length,
        model_card_data=CrossEncoderModelCardData(
            language=language,
            license=license,
            model_name=model_card_name or f"{short_model_name} trained on custom dataset",
        ),
        model_kwargs={"attn_implementation": "flash_attention_2"},
    )
    logger.info("Model max length: %s, num labels: %s", model.max_length, model.num_labels)
    _log_rss("after model load")

    logger.info("Loading bucketed training dataset from: %s", train_dataset_path)
    if (Path(train_dataset_path) / "manifest.json").exists() or (Path(train_dataset_path).parent / "manifest.json").exists():
        train_dataset, train_groups = load_bucketed_split(
            train_dataset_path,
            split="train",
            rank=rank,
            world_size=world_size,
            assignment=bucket_assignment,
        )
    else:
        logger.warning("Using fallback plain parquet loader. For DDP memory safety, use prepare_buckets.py output.")
        train_dataset, train_groups = load_plain_parquets_with_groups(train_dataset_path)

    if len(train_dataset) == 0:
        raise RuntimeError(f"Rank {rank} has empty training dataset. Increase num_buckets or change assignment.")
    logger.info("Rank %d training dataset size: %d", rank, len(train_dataset))
    _log_rss("after train dataset load")

    index_to_anchor, index_to_positive_key = build_key_maps(train_dataset, no_duplicates=no_duplicates)
    training_args.batch_sampler = make_same_dataset_batch_sampler(
        train_groups,
        no_duplicates=no_duplicates,
        index_to_anchor=index_to_anchor,
        index_to_positive_key=index_to_positive_key,
        repeat_for_accelerate_sharding=False,
    )
    _log_rss("after sampler key map build")

    # Critical: do not pass bucket/helper/source columns into CrossEncoderTrainer.
    # The sampler has already consumed query_key64 / positive_key64 above.
    # Leaving numeric columns such as bucket_id/query_key64 in train_dataset can make
    # SentenceTransformers treat them as text fields and pass ints to tokenizer.
    train_dataset = sanitize_training_dataset_for_loss(train_dataset, num_hard_negatives=num_hard_negatives)
    gc.collect()
    _log_rss("after sanitizing train dataset columns")

    logger.info("Loading bucketed evaluation dataset from: %s", eval_dataset_path)
    if (Path(eval_dataset_path) / "manifest.json").exists() or (Path(eval_dataset_path).parent / "manifest.json").exists():
        eval_dataset, _ = load_bucketed_split(
            eval_dataset_path,
            split="validation",
            rank=rank,
            world_size=world_size,
            assignment=bucket_assignment,
        )
    else:
        eval_dataset, _ = load_plain_parquets_with_groups(eval_dataset_path)

    logger.info("Rank %d evaluation dataset size: %d", rank, len(eval_dataset))
    _log_rss("after eval dataset load")

    eval_samples = prepare_reranking_samples(eval_dataset, num_negatives)
    evaluator = Evaluator(
        samples=eval_samples,
        name="custom-dev",
        batch_size=training_args.per_device_eval_batch_size,
        show_progress_bar=False,
    )
    logger.info("Initialized evaluator with %d local samples on rank %d", len(eval_samples), rank)

    del eval_dataset
    gc.collect()
    _log_rss("after deleting eval_dataset")

    logger.info(
        "Initializing loss with %d hard negatives, %d random negatives, scale=%.3f",
        num_negatives,
        num_random_negatives,
        loss_scale,
    )
    loss = PositiveOnlyInBatchCachedMultipleNegativesRankingLoss(
        mini_batch_size=mini_batch_size,
        model=model,
        num_negatives=num_random_negatives,
        scale=loss_scale,
        negative_sampling_seed=seed + rank,
    )

    logger.info("Initializing CrossEncoderTrainer")
    trainer = UnshardedCrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        loss=loss,
        evaluator=evaluator,
        callbacks=[
            TrainingPhaseTimingCallback(),
            LogitRangeCallback(model, eval_samples),
        ],
    )
    train_dataloader = trainer.get_train_dataloader()
    train_dataloader_len = len(train_dataloader)
    dataloader_sampler = getattr(train_dataloader, "batch_sampler", None)
    schedule_sampler = _find_sampler_with_schedule_len(train_dataloader)
    expected_schedule_len = getattr(schedule_sampler, "_last_schedule_len", None)
    logger.info(
        "Train dataloader length after Accelerate config: %s (batch_sampler=%s schedule_sampler=%s expected_schedule_len=%s)",
        train_dataloader_len,
        type(dataloader_sampler).__name__ if dataloader_sampler is not None else None,
        type(schedule_sampler).__name__ if schedule_sampler is not None else None,
        expected_schedule_len,
    )
    if expected_schedule_len is not None and train_dataloader_len != expected_schedule_len:
        raise RuntimeError(
            f"Accelerate appears to be sharding or padding the custom batch sampler: "
            f"len(dataloader)={train_dataloader_len}, synchronized_schedule_len={expected_schedule_len}"
        )

    resume_from_checkpoint = _resolve_resume_checkpoint(training_args.output_dir)
    if resume_from_checkpoint:
        logger.info("Resuming training from checkpoint: %s", resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        logger.info("Starting training from scratch (set RESUME_FROM_CHECKPOINT=latest to resume)")
        trainer.train()

    logger.info("Running final evaluation...")
    eval_results = evaluator(model)
    metric_name = _evaluator_metric_name(training_args.metric_for_best_model)
    logger.info("Final %s: %.4f", metric_name, eval_results.get(metric_name, float("nan")))

    if use_clearml and task and _is_global_rank_zero():
        task.get_logger().report_scalar(
            title="Evaluation",
            series=metric_name,
            value=eval_results.get(metric_name, float("nan")),
            iteration=training_args.num_train_epochs,
        )

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to: %s", output_dir)
    model.save_pretrained(str(output_dir / "final_model"))

    if use_clearml and task and _is_global_rank_zero():
        task.close()

    return model


def main():
    training_args = CrossEncoderTrainingArguments(
        output_dir="/home/jovyan/new-pvc/cross_encoders/PositiveOnly-RuModernBERT-base-reranker-training-hneg8-rneg2-cosine-lr1e5-minibatch128-pgpubatch256-gpu8-cf-no-pin-workers2/",
        num_train_epochs=3,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=64,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        tf32=True,
        eval_strategy="steps",
        eval_steps=189,
        save_strategy="steps",
        save_steps=189,
        save_total_limit=5,
        logging_steps=5,
        logging_first_step=True,
        disable_tqdm=True,
        run_name="RuModernBERT-base-reranker-training-bucketed",
        seed=12,
        report_to="clearml",
        optim="adamw_torch_fused",
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_custom-dev_map",
        greater_is_better=True,
        accelerator_config={
            "even_batches": False,
            "dispatch_batches": False,
            "split_batches": False,
            "use_seedable_sampler": False,
        },
    )

    model = train_cross_encoder_reranker(
        model_name_or_path="/home/jovyan/new-pvc/cross_encoders/nik_datasets/models/RuModernBERT-base",
        train_dataset_path="/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/final_dataset_bucketed_v6/train/",
        eval_dataset_path="/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/final_dataset_bucketed_v6/validation/",
        mini_batch_size=128,
        training_args=training_args,
        num_hard_negatives=8,
        num_negatives=8,
        num_random_negatives=2,
        clearml_task_name="PositiveOnly-RuModernBERT-base-reranker-training-hneg8-rneg2-cosine-lr1e5-minibatch128-pgpubatch256-gpu8-cf-no-pin-workers2",
        max_length=512,
        no_duplicates=True,
        bucket_assignment="greedy",
        loss_scale=10.0,
    )

    logger.info("Training completed successfully: %s", model)


if __name__ == "__main__":
    main()
