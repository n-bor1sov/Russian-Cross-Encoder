
from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _rank_from_env_for_logging() -> int:
    for env in ("LOCAL_RANK", "RANK", "SLURM_LOCALID"):
        value = os.environ.get(env)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def _show_batch_sampler_progress() -> bool:
    """Whether tqdm bars are allowed.

    Disabled by default because tqdm writes frequently to stdout/stderr and can
    block distributed jobs when launcher/ClearML/tee pipes fill. Set
    ENABLE_TQDM=1 for short interactive debugging.
    """
    if _env_flag("ENABLE_TQDM", "0"):
        return True
    if _env_flag("DISABLE_TQDM", "1"):
        return False
    if _env_flag("SHOW_ALL_TQDM", "0"):
        return True
    return _rank_from_env_for_logging() == 0


def _log_batch_sampler_progress() -> bool:
    """Whether this rank should emit periodic sampler INFO logs."""
    if _env_flag("SHOW_ALL_RANK_LOGS", "0"):
        return True
    return _rank_from_env_for_logging() == 0

def _dist_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    try:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except ValueError:
        rank = 0
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    return rank, world_size


class SetEpochMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SameDatasetBatchSampler(SetEpochMixin):
    """Dataset-homogeneous DDP-safe batch sampler.

    The sampler receives already-local row indices. It does not perform rank
    sharding internally. Instead, it builds local batches for every dataset,
    synchronizes the per-dataset number of available batches across ranks with
    all_reduce(MIN), builds the same shuffled dataset schedule on every rank,
    and yields exactly one local batch per scheduled dataset per global step.

    Accelerate sharding should be disabled in TrainingArguments via
    accelerator_config, because the dataset is already rank-local. This avoids
    the previous rank-local chunk shuffle that can make different ranks train on
    different datasets at the same DDP step.
    """

    def __init__(
        self,
        groups: dict[str, list[int]],
        batch_size: int,
        drop_last: bool,
        num_processes: int | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
        no_duplicates: bool = False,
        index_to_anchor: dict[int, int] | None = None,
        index_to_positive_key: dict[int, int] | None = None,
        repeat_for_accelerate_sharding: bool = False,
    ) -> None:
        super().__init__()
        self.groups = {name: list(indices) for name, indices in groups.items()}
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        rank, world_size = _dist_info()
        self.rank = rank
        self.num_processes = num_processes or world_size
        self.generator = generator
        self.seed = int(seed)
        self.no_duplicates = bool(no_duplicates)
        self.index_to_anchor = index_to_anchor or {}
        self.index_to_positive_key = index_to_positive_key or {}
        self.repeat_for_accelerate_sharding = bool(repeat_for_accelerate_sharding)
        self._repeat_factor = self.num_processes if (self.repeat_for_accelerate_sharding and self.num_processes > 1) else 1
        self._last_schedule_len: int | None = None
        self._last_raw_len: int | None = None
        self._last_global_counts: dict[str, int] | None = None

        logger.info(
            "SameDatasetBatchSampler rank=%d/%d: %d dataset(s), batch_size=%d, "
            "drop_last=%s, no_duplicates=%s, repeat_for_accelerate_sharding=%s, "
            "repeat_factor=%d, rows_by_dataset=%s",
            self.rank,
            self.num_processes,
            len(self.groups),
            self.batch_size,
            self.drop_last,
            self.no_duplicates,
            self.repeat_for_accelerate_sharding,
            self._repeat_factor,
            {k: len(v) for k, v in self.groups.items()},
        )
        self._num_batches = self._compute_num_batches_local()

    def _compute_num_batches_local(self) -> int:
        total = 0
        for indices in self.groups.values():
            n = len(indices)
            total += n // self.batch_size
            if not self.drop_last and n % self.batch_size > 0:
                total += 1
        return total

    def _compute_num_batches_by_dataset_local(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for dataset_name, indices in self.groups.items():
            n = len(indices)
            count = n // self.batch_size
            if not self.drop_last and n % self.batch_size > 0:
                count += 1
            if count > 0:
                counts[dataset_name] = count
        return counts

    def _shuffled_indices(self, indices: list[int]) -> list[int]:
        if not indices:
            return []
        perm = torch.randperm(len(indices), generator=self.generator)
        return [indices[i] for i in perm.tolist()]

    def _build_plain_batches(self, indices: list[int]) -> list[list[int]]:
        shuffled = self._shuffled_indices(indices)
        batches: list[list[int]] = []
        for start in range(0, len(shuffled), self.batch_size):
            batch = shuffled[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            batches.append(batch)
        return batches

    def _build_no_duplicate_batches(
        self,
        indices: list[int],
        dataset_name: str,
        show_progress: bool,
    ) -> list[list[int]]:
        """Build batches with no duplicate query-family key or positive key.

        Keys are expected to be stable 64-bit integers (query_key64 and
        positive_key64) whenever the bucketed dataset was prepared by
        prepare_buckets.py.
        """
        anchor_to_q: dict[int, deque[int]] = {}
        row_iter = indices
        if show_progress:
            row_iter = tqdm(
                indices,
                desc=f"No-dup '{dataset_name}': rows→anchors",
                unit="row",
                leave=False,
            )

        for idx in row_iter:
            anchor_key = self.index_to_anchor.get(idx)
            if anchor_key is None:
                anchor_key = -1 - int(idx)
            anchor_to_q.setdefault(int(anchor_key), deque()).append(idx)

        for group in anchor_to_q.values():
            items = list(group)
            if len(items) > 1:
                perm = torch.randperm(len(items), generator=self.generator)
                items = [items[i] for i in perm.tolist()]
            group.clear()
            group.extend(items)

        groups = list(anchor_to_q.values())
        if len(groups) > 1:
            perm = torch.randperm(len(groups), generator=self.generator)
            groups = [groups[i] for i in perm.tolist()]

        batches: list[list[int]] = []
        batch_bar = tqdm(
            desc=f"No-dup '{dataset_name}': form batches",
            unit="batch",
            leave=False,
        ) if show_progress else None

        try:
            while groups:
                batch: list[int] = []
                used_positive_keys: set[int] = set()
                next_round: list[deque[int]] = []
                made_progress = False

                for group in groups:
                    if len(batch) >= self.batch_size:
                        next_round.append(group)
                        continue

                    idx = group[0]
                    pos_key = self.index_to_positive_key.get(idx)
                    if pos_key is None:
                        pos_key = -10_000_000_000 - int(idx)

                    if int(pos_key) in used_positive_keys:
                        next_round.append(group)
                        continue

                    idx = group.popleft()
                    batch.append(idx)
                    used_positive_keys.add(int(pos_key))
                    made_progress = True
                    if group:
                        next_round.append(group)

                if not made_progress:
                    raise RuntimeError(
                        f"No progress while building no-duplicate batch for dataset={dataset_name}. "
                        f"Remaining anchor groups={len(groups)}. This usually means too many rows "
                        "share the same positive_key64 for the current no-duplicate constraints."
                    )

                if len(batch) < self.batch_size and self.drop_last:
                    break

                batches.append(batch)
                groups = next_round
                if batch_bar is not None:
                    batch_bar.update(1)
        finally:
            if batch_bar is not None:
                batch_bar.close()

        return batches

    def _synchronize_counts(self, dataset_names: list[str], group_batches: dict[str, list[list[int]]]) -> list[int]:
        local_counts = torch.tensor(
            [len(group_batches[name]) for name in dataset_names],
            dtype=torch.long,
        )

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            device = torch.device(
                f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            )
            min_counts = local_counts.to(device)
            sum_counts = local_counts.to(device)
            dist.all_reduce(min_counts, op=dist.ReduceOp.MIN)
            dist.all_reduce(sum_counts, op=dist.ReduceOp.SUM)
            global_counts = min_counts.cpu().tolist()
            mean_counts = [float(count) / dist.get_world_size() for count in sum_counts.cpu().tolist()]
        else:
            global_counts = local_counts.tolist()
            mean_counts = [float(x) for x in global_counts]

        self._last_global_counts = {name: int(count) for name, count in zip(dataset_names, global_counts)}
        if _log_batch_sampler_progress():
            logger.info(
                "SameDatasetBatchSampler rank=%d: synchronized batch counts local=%s min=%s mean=%s",
                self.rank,
                {name: int(count) for name, count in zip(dataset_names, local_counts.tolist())},
                self._last_global_counts,
                {name: float(count) for name, count in zip(dataset_names, mean_counts)},
            )

        bad_counts = [
            (name, int(min_count), float(mean_count))
            for name, min_count, mean_count in zip(dataset_names, global_counts, mean_counts)
            if mean_count > 0 and (int(min_count) == 0 or int(min_count) < max(1, math.ceil(0.5 * mean_count)))
        ]
        if bad_counts and not _env_flag("ALLOW_UNBALANCED_DATASET_COUNTS", "0"):
            raise RuntimeError(
                "Synchronized sampler would drop or heavily under-sample dataset(s) due to uneven rank-local "
                f"batch counts: {bad_counts}. Use bucket_assignment='greedy', increase buckets, or set "
                "ALLOW_UNBALANCED_DATASET_COUNTS=1 to override."
            )
        return [int(x) for x in global_counts]

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        show_tqdm = _show_batch_sampler_progress()
        show_logs = _log_batch_sampler_progress()
        use_no_dup_build = self.no_duplicates and bool(self.index_to_anchor)

        group_batches: dict[str, list[list[int]]] = {}
        group_items = [(k, v) for k, v in self.groups.items() if len(v) > 0]

        if show_logs:
            logger.info(
                "SameDatasetBatchSampler rank=%d: building local batch lists for %d rows in %d dataset group(s)",
                self.rank,
                sum(len(v) for _, v in group_items),
                len(group_items),
            )

        group_iter = tqdm(
            group_items,
            desc="Sampler: per-dataset local batch build",
            unit="dataset",
            leave=True,
            disable=not show_tqdm,
        )

        for dataset_name, indices in group_iter:
            dataset_t0 = time.perf_counter()
            if use_no_dup_build:
                batches = self._build_no_duplicate_batches(
                    indices,
                    dataset_name=dataset_name,
                    show_progress=show_tqdm,
                )
            else:
                batches = self._build_plain_batches(indices)
            group_batches[dataset_name] = batches
            if show_logs:
                logger.info(
                    "SameDatasetBatchSampler rank=%d: built dataset=%s rows=%d batches=%d in %.1fs",
                    self.rank,
                    dataset_name,
                    len(indices),
                    len(batches),
                    time.perf_counter() - dataset_t0,
                )

        dataset_names = sorted(group_batches.keys())
        global_counts = self._synchronize_counts(dataset_names, group_batches)

        for dataset_name, count in zip(dataset_names, global_counts):
            group_batches[dataset_name] = group_batches[dataset_name][:count]

        schedule: list[str] = []
        for dataset_name, count in zip(dataset_names, global_counts):
            if count > 0:
                schedule.extend([dataset_name] * count)

        if not schedule:
            raise RuntimeError(
                "SameDatasetBatchSampler produced an empty synchronized schedule. "
                f"rank={self.rank}, local_counts="
                f"{ {name: len(group_batches.get(name, [])) for name in dataset_names} }, "
                f"global_counts={dict(zip(dataset_names, global_counts))}"
            )

        perm = torch.randperm(len(schedule), generator=self.generator)
        schedule = [schedule[i] for i in perm.tolist()]
        self._last_schedule_len = len(schedule)

        raw_len = len(schedule) * self._repeat_factor
        self._last_raw_len = raw_len

        if show_logs:
            logger.info(
                "SameDatasetBatchSampler rank=%d: synchronized schedule has %d effective steps; "
                "raw sampler length=%d; repeat_factor=%d; counts=%s",
                self.rank,
                len(schedule),
                raw_len,
                self._repeat_factor,
                dict(zip(dataset_names, global_counts)),
            )

        cursors = {name: 0 for name in dataset_names}
        step_iter = tqdm(
            schedule,
            desc="Sampler: synchronized dataset schedule",
            unit="step",
            leave=False,
            disable=not show_tqdm,
        )

        for step_idx, dataset_name in enumerate(step_iter):
            batch_idx = cursors[dataset_name]
            cursors[dataset_name] += 1

            if show_logs and step_idx % 100 == 0:
                logger.info(
                    "Yielding synchronized effective_step=%d dataset=%s local_batch_idx=%d repeat_factor=%d",
                    step_idx,
                    dataset_name,
                    batch_idx,
                    self._repeat_factor,
                )

            batch = group_batches[dataset_name][batch_idx]

            for _ in range(self._repeat_factor):
                yield batch

    def __len__(self) -> int:
        # With Accelerate batch-sampler sharding disabled, this is the actual
        # synchronized schedule length. The fallback is only approximate before
        # the first cross-rank count synchronization.
        if self._last_raw_len is not None:
            return self._last_raw_len
        if self._last_schedule_len is not None:
            return self._last_schedule_len * self._repeat_factor
        local_counts_by_dataset = self._compute_num_batches_by_dataset_local()
        dataset_names = sorted(local_counts_by_dataset)
        if not dataset_names:
            return 0
        local_counts = torch.tensor(
            [local_counts_by_dataset[name] for name in dataset_names],
            dtype=torch.long,
        )
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            device = torch.device(
                f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            )
            counts = local_counts.to(device)
            dist.all_reduce(counts, op=dist.ReduceOp.MIN)
            schedule_len = int(counts.cpu().sum().item())
            self._last_schedule_len = schedule_len
            self._last_raw_len = schedule_len * self._repeat_factor
            return self._last_raw_len
        else:
            schedule_len = int(local_counts.sum().item())
            if self.num_processes <= 1:
                self._last_schedule_len = schedule_len
                self._last_raw_len = schedule_len * self._repeat_factor
            return schedule_len * self._repeat_factor


class SameDatasetBatchSamplerFactory:
    """Picklable factory for a pre-grouped local dataset.

    The dataset must already contain only rows assigned to the current rank.
    """

    def __init__(
        self,
        groups: dict[str, list[int]],
        no_duplicates: bool = False,
        index_to_anchor: dict[int, int] | None = None,
        index_to_positive_key: dict[int, int] | None = None,
        repeat_for_accelerate_sharding: bool = False,
    ) -> None:
        self.groups = {k: list(v) for k, v in groups.items()}
        self.no_duplicates = bool(no_duplicates)
        self.index_to_anchor = dict(index_to_anchor or {})
        self.index_to_positive_key = dict(index_to_positive_key or {})
        self.repeat_for_accelerate_sharding = bool(repeat_for_accelerate_sharding)

    def __call__(
        self,
        dataset: Any,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> SameDatasetBatchSampler:
        return SameDatasetBatchSampler(
            groups=self.groups,
            batch_size=batch_size,
            drop_last=drop_last,
            generator=generator,
            seed=seed,
            no_duplicates=self.no_duplicates,
            index_to_anchor=self.index_to_anchor,
            index_to_positive_key=self.index_to_positive_key,
            repeat_for_accelerate_sharding=self.repeat_for_accelerate_sharding,
        )


def make_same_dataset_batch_sampler(
    groups: dict[str, list[int]],
    no_duplicates: bool = False,
    index_to_anchor: dict[int, int] | None = None,
    index_to_positive_key: dict[int, int] | None = None,
    repeat_for_accelerate_sharding: bool = False,
) -> SameDatasetBatchSamplerFactory:
    """Build a sampler factory for a rank-local bucketed dataset."""
    return SameDatasetBatchSamplerFactory(
        groups=groups,
        no_duplicates=no_duplicates,
        index_to_anchor=index_to_anchor,
        index_to_positive_key=index_to_positive_key,
        repeat_for_accelerate_sharding=repeat_for_accelerate_sharding,
    )
