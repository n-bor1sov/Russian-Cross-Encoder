# Repo Paper-Readability Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the CE_repo (Russian cross-encoder reranker) into the documented six-stage layout from `docs/superpowers/specs/2026-05-26-repo-paper-readability-design.md`, with per-stage TOML configs, shell runners, narrative documentation, and a self-contained examples directory — without changing algorithm behavior.

**Architecture:** File reorganization via `git mv` (history preserved), inlined `--config` TOML loading per script (no shared Python package), `.env` for paths, shell runners that wrap each Python entrypoint, narrative documentation in `docs/` with self-contained per-stage READMEs. Unification moves out of the pipeline into `data_prep/`.

**Tech Stack:** Python 3.13 (`tomllib` from stdlib), `uv` for dependency management, Bash shell runners, Parquet (`pyarrow`) for IO, existing dependencies preserved.

---

## Notes for the implementer

- Read the design spec at `docs/superpowers/specs/2026-05-26-repo-paper-readability-design.md` before starting. Tasks below assume that spec as ground truth.
- **All file moves use `git mv`** — never `mv` followed by `git add`. This preserves blame history.
- **No algorithm changes.** The only behavior change permitted is fixing `restore_only.py`'s broken `runpy` indirection.
- **No tests.** Verification is `python -c "import ast; ast.parse(open(p).read())"` for syntax, `--help` for CLI sanity, and `examples/run.sh` end-to-end at the end.
- **No thesis section references in user-facing docs.** Equation numbers inside code comments are allowed; `§3.X`, "Chapter 3", "Methodology chapter" in markdown is not.
- The repo currently has only one commit (the design spec itself). Existing source files are untracked. **Task 1** commits them as-is so subsequent moves preserve history.
- `multi_slerp` does **not** currently exist in the codebase. The plan writes `tools/slerp_merge.py` from scratch (Task 18).
- `dataset_scripts/debug_faiss_gpu.py` is not currently present in the working tree, so no deletion is needed.

---

## Task 1: Commit existing source as a baseline

**Files:**
- Add all untracked files to git.

- [ ] **Step 1: Confirm current untracked state**

Run: `git status -s`
Expected output includes untracked entries like `?? dataset_scripts/`, `?? train_scripts/`, `?? benchmarking/`, `?? README.md`, `?? pyproject.toml`, `?? uv.lock`, `?? .env.example`, `?? .gitignore`, `?? .python-version`, `?? thesis.pdf`.

- [ ] **Step 2: Stage all existing files**

```bash
git add .env.example .gitignore .python-version README.md pyproject.toml uv.lock thesis.pdf
git add dataset_scripts/ train_scripts/ benchmarking/ .claude/
```

- [ ] **Step 3: Verify staging**

Run: `git status -s`
Expected: every previously-untracked file now appears as `A` (added).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: import existing pipeline source as baseline

Snapshot of the working scripts before the paper-readability restructure.
No content changes."
```

---

## Task 2: Add MIT LICENSE file

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Write the LICENSE file**

Create `LICENSE` with the standard MIT text:

```
MIT License

Copyright (c) 2026 Borisov Nikita Mikhailovich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Stage and commit**

```bash
git add LICENSE
git commit -m "chore: add MIT license"
```

---

## Task 3: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append `examples/work/` to .gitignore**

Append the following lines to `.gitignore`:

```
# Local example pipeline outputs
examples/work/
```

- [ ] **Step 2: Verify**

Run: `tail -3 .gitignore`
Expected: ends with the two lines added above.

- [ ] **Step 3: Stage and commit**

```bash
git add .gitignore
git commit -m "chore: gitignore examples/work/"
```

---

## Task 4: Create new directory skeleton

**Files:**
- Create empty directories. Git does not track empty directories, so each will get a `.gitkeep` placeholder that is removed in later tasks once real files arrive.

- [ ] **Step 1: Create all target directories**

```bash
mkdir -p data_prep
mkdir -p pipeline/01_split
mkdir -p pipeline/02_embed
mkdir -p pipeline/03_filter_consistency
mkdir -p pipeline/04_hard_negative_mining/slurm
mkdir -p pipeline/05_compile
mkdir -p pipeline/06_train
mkdir -p evaluation
mkdir -p tools
mkdir -p configs/thesis
mkdir -p docs
mkdir -p examples/data
mkdir -p examples/configs
mkdir -p examples/expected_outputs
```

- [ ] **Step 2: Verify**

Run: `find . -type d -name 'pipeline' -o -name 'data_prep' -o -name 'evaluation' -o -name 'tools' -o -name 'examples' | head`
Expected: all five top-level new directories exist.

No commit yet — directories are committed implicitly when files are moved into them.

---

## Task 5: Move files to `data_prep/`

**Files:**
- `dataset_scripts/01_merge_to_shards.py` → `data_prep/merge_to_shards.py`
- `dataset_scripts/add_parquet_dataset_column.py` → `data_prep/add_dataset_column.py`

- [ ] **Step 1: Move the two scripts with git mv**

```bash
git mv dataset_scripts/01_merge_to_shards.py data_prep/merge_to_shards.py
git mv dataset_scripts/add_parquet_dataset_column.py data_prep/add_dataset_column.py
```

- [ ] **Step 2: Verify**

Run: `ls data_prep/`
Expected: `add_dataset_column.py  merge_to_shards.py`

Run: `python -c "import ast; [ast.parse(open(p).read()) for p in ['data_prep/merge_to_shards.py', 'data_prep/add_dataset_column.py']]"`
Expected: no output (no syntax error).

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: move unification helpers into data_prep/"
```

---

## Task 6: Move files to `pipeline/01_split/`

**Files:**
- `dataset_scripts/split_datasets.py` → `pipeline/01_split/split.py`

- [ ] **Step 1: Move**

```bash
git mv dataset_scripts/split_datasets.py pipeline/01_split/split.py
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('pipeline/01_split/split.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: move split_datasets.py to pipeline/01_split/split.py"
```

---

## Task 7: Move files to `pipeline/02_embed/`

**Files:**
- `dataset_scripts/02_embed_shards.py` → `pipeline/02_embed/embed_shards.py`

- [ ] **Step 1: Move**

```bash
git mv dataset_scripts/02_embed_shards.py pipeline/02_embed/embed_shards.py
```

- [ ] **Step 2: Verify**

Run: `python -c "import ast; ast.parse(open('pipeline/02_embed/embed_shards.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: move embed_shards.py to pipeline/02_embed/"
```

---

## Task 8: Move files to `pipeline/03_filter_consistency/` and delete old variant

**Files:**
- Delete: `dataset_scripts/03_filter_and_restore_datasets.py`
- `dataset_scripts/03_filter_and_restore_datasets_fixed.py` → `pipeline/03_filter_consistency/score.py`
- `dataset_scripts/04_restore_embeddings_by_dataset.py` → `pipeline/03_filter_consistency/restore_embeddings.py`
- `dataset_scripts/05_filter_by_rank.py` → `pipeline/03_filter_consistency/filter_by_rank.py`
- `dataset_scripts/restore_only_datasets.py` → `pipeline/03_filter_consistency/restore_only.py`

- [ ] **Step 1: Delete the superseded variant**

```bash
git rm dataset_scripts/03_filter_and_restore_datasets.py
```

- [ ] **Step 2: Move the canonical variant and the related scripts**

```bash
git mv dataset_scripts/03_filter_and_restore_datasets_fixed.py pipeline/03_filter_consistency/score.py
git mv dataset_scripts/04_restore_embeddings_by_dataset.py pipeline/03_filter_consistency/restore_embeddings.py
git mv dataset_scripts/05_filter_by_rank.py pipeline/03_filter_consistency/filter_by_rank.py
git mv dataset_scripts/restore_only_datasets.py pipeline/03_filter_consistency/restore_only.py
```

- [ ] **Step 3: Verify**

Run: `ls pipeline/03_filter_consistency/`
Expected: `filter_by_rank.py  restore_embeddings.py  restore_only.py  score.py`

Run: `python -c "import ast; [ast.parse(open(p).read()) for p in ['pipeline/03_filter_consistency/score.py','pipeline/03_filter_consistency/restore_embeddings.py','pipeline/03_filter_consistency/filter_by_rank.py']]"`
Expected: no output. (`restore_only.py` is expected to fail — it references the now-deleted file. Task 14 fixes it.)

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: consolidate consistency filtering into pipeline/03_filter_consistency/

The _fixed variant becomes the canonical score.py; the older variant is deleted.
restore_only.py is broken at this commit (still imports the deleted file via runpy)
and is repaired in a follow-up task."
```

---

## Task 9: Move files to `pipeline/04_hard_negative_mining/`

**Files:**
- `dataset_scripts/hnm/HNM.py` → `pipeline/04_hard_negative_mining/mine.py`
- `dataset_scripts/hnm/embedding_lookup.py` → `pipeline/04_hard_negative_mining/embedding_lookup.py`
- `dataset_scripts/hnm/HNM.sh` → `pipeline/04_hard_negative_mining/slurm/HNM.sh`
- `dataset_scripts/hnm/HNM_live.sh` → `pipeline/04_hard_negative_mining/slurm/HNM_live.sh`
- `dataset_scripts/hnm/kill_hnm.sh` → `pipeline/04_hard_negative_mining/slurm/kill_hnm.sh`
- `dataset_scripts/hnm/status_hnm.sh` → `pipeline/04_hard_negative_mining/slurm/status_hnm.sh`

- [ ] **Step 1: Move Python files**

```bash
git mv dataset_scripts/hnm/HNM.py pipeline/04_hard_negative_mining/mine.py
git mv dataset_scripts/hnm/embedding_lookup.py pipeline/04_hard_negative_mining/embedding_lookup.py
```

- [ ] **Step 2: Move shell launchers into slurm/ subdir**

```bash
git mv dataset_scripts/hnm/HNM.sh pipeline/04_hard_negative_mining/slurm/HNM.sh
git mv dataset_scripts/hnm/HNM_live.sh pipeline/04_hard_negative_mining/slurm/HNM_live.sh
git mv dataset_scripts/hnm/kill_hnm.sh pipeline/04_hard_negative_mining/slurm/kill_hnm.sh
git mv dataset_scripts/hnm/status_hnm.sh pipeline/04_hard_negative_mining/slurm/status_hnm.sh
```

- [ ] **Step 3: Verify**

Run: `ls pipeline/04_hard_negative_mining/ pipeline/04_hard_negative_mining/slurm/`
Expected:
- `pipeline/04_hard_negative_mining/`: `embedding_lookup.py  mine.py  slurm`
- `pipeline/04_hard_negative_mining/slurm/`: `HNM.sh  HNM_live.sh  kill_hnm.sh  status_hnm.sh`

Run: `python -c "import ast; [ast.parse(open(p).read()) for p in ['pipeline/04_hard_negative_mining/mine.py','pipeline/04_hard_negative_mining/embedding_lookup.py']]"`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: move HNM code to pipeline/04_hard_negative_mining/

Python entrypoint becomes mine.py; cluster launchers go into slurm/.
Slurm scripts still reference the old paths and are fixed in a follow-up."
```

---

## Task 10: Move files to `pipeline/05_compile/`

**Files:**
- `dataset_scripts/finale/final_dataset_compilation.py` → `pipeline/05_compile/compile.py`
- `dataset_scripts/finale/analyze_final_dataset.py` → `pipeline/05_compile/analyze.py`

- [ ] **Step 1: Move**

```bash
git mv dataset_scripts/finale/final_dataset_compilation.py pipeline/05_compile/compile.py
git mv dataset_scripts/finale/analyze_final_dataset.py pipeline/05_compile/analyze.py
```

- [ ] **Step 2: Verify**

Run: `python -c "import ast; [ast.parse(open(p).read()) for p in ['pipeline/05_compile/compile.py','pipeline/05_compile/analyze.py']]"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: move final dataset compilation to pipeline/05_compile/"
```

---

## Task 11: Move files to `pipeline/06_train/`

**Files:**
- `train_scripts/prepare_query_buckets.py` → `pipeline/06_train/prepare_buckets.py`
- `train_scripts/train_bucketed.py` → `pipeline/06_train/train.py`
- `train_scripts/sampler_bucketed.py` → `pipeline/06_train/sampler.py`

- [ ] **Step 1: Move**

```bash
git mv train_scripts/prepare_query_buckets.py pipeline/06_train/prepare_buckets.py
git mv train_scripts/train_bucketed.py pipeline/06_train/train.py
git mv train_scripts/sampler_bucketed.py pipeline/06_train/sampler.py
```

- [ ] **Step 2: Verify**

Run: `python -c "import ast; [ast.parse(open(p).read()) for p in ['pipeline/06_train/prepare_buckets.py','pipeline/06_train/train.py','pipeline/06_train/sampler.py']]"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: move training scripts to pipeline/06_train/"
```

---

## Task 12: Move `profile_training.py` to `tools/`

**Files:**
- `train_scripts/profile_training.py` → `tools/profile_training.py`

- [ ] **Step 1: Move**

```bash
git mv train_scripts/profile_training.py tools/profile_training.py
```

- [ ] **Step 2: Verify**

Run: `python -c "import ast; ast.parse(open('tools/profile_training.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: move profile_training.py to tools/"
```

---

## Task 13: Move evaluation scripts to `evaluation/`

**Files:**
- `benchmarking/generate_bm25_top100.py` → `evaluation/generate_bm25_top100.py`
- `benchmarking/evaluate_reranker_top100.py` → `evaluation/evaluate_reranker_top100.py`

- [ ] **Step 1: Move**

```bash
git mv benchmarking/generate_bm25_top100.py evaluation/generate_bm25_top100.py
git mv benchmarking/evaluate_reranker_top100.py evaluation/evaluate_reranker_top100.py
```

- [ ] **Step 2: Verify and confirm old directories are empty**

Run: `ls evaluation/`
Expected: `evaluate_reranker_top100.py  generate_bm25_top100.py`

Run: `ls dataset_scripts/ train_scripts/ benchmarking/ 2>&1 | head`
Expected: each is either empty or shows only sub-directories (`dataset_scripts/hnm`, `dataset_scripts/finale`) that are also empty.

- [ ] **Step 3: Remove empty source directories**

```bash
# rmdir fails if the directory has anything left — that's the safety net
rmdir dataset_scripts/hnm dataset_scripts/finale dataset_scripts
rmdir train_scripts
rmdir benchmarking
```

If any rmdir fails, list the directory contents and stop — there's an unmigrated file that must be addressed before continuing.

- [ ] **Step 4: Verify cleanup**

Run: `[ ! -d dataset_scripts ] && [ ! -d train_scripts ] && [ ! -d benchmarking ] && echo OK`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: move evaluation scripts to evaluation/ and drop old top-level dirs

dataset_scripts/, train_scripts/, and benchmarking/ are gone; all content has
migrated to data_prep/, pipeline/, tools/, and evaluation/."
```

---

## Task 14: Fix `restore_only.py` runpy dependency

**Files:**
- Modify: `pipeline/03_filter_consistency/restore_only.py`

Current state: this script uses `runpy.run_path` to load functions from the now-deleted `03_filter_and_restore_datasets.py`. After Task 8 the import target is gone. The fix is to replace `runpy` with a normal sibling import from `score.py`.

- [ ] **Step 1: Inspect the current top of restore_only.py**

Run: `head -20 pipeline/03_filter_consistency/restore_only.py`

Expected to show:
```python
MODULE_PATH = Path(__file__).with_name("03_filter_and_restore_datasets.py")
MODULE = runpy.run_path(str(MODULE_PATH))
restore_dataset_outputs = MODULE["restore_dataset_outputs"]
validate_dataset_outputs = MODULE["validate_dataset_outputs"]
```

- [ ] **Step 2: Confirm the symbols exist in score.py**

Run: `grep -n "def restore_dataset_outputs\|def validate_dataset_outputs" pipeline/03_filter_consistency/score.py`
Expected: both function definitions exist.

If either is missing, halt and inspect `score.py` to find the renamed equivalents. Update Step 3 below to import the correct names.

- [ ] **Step 3: Replace the runpy block with a normal sibling import**

Replace the top of `pipeline/03_filter_consistency/restore_only.py` so the imports use a `sys.path` shim (the script is invoked directly, so it's not part of a package):

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow sibling import when invoked as `python pipeline/03_filter_consistency/restore_only.py ...`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import restore_dataset_outputs, validate_dataset_outputs  # noqa: E402
```

Delete the now-unused `runpy` import.

- [ ] **Step 4: Syntax check**

Run: `python -c "import ast; ast.parse(open('pipeline/03_filter_consistency/restore_only.py').read())"`
Expected: no output.

- [ ] **Step 5: CLI sanity check**

Run: `python pipeline/03_filter_consistency/restore_only.py --help`
Expected: argparse prints the usage block and exits cleanly. No traceback.

- [ ] **Step 6: Commit**

```bash
git add pipeline/03_filter_consistency/restore_only.py
git commit -m "fix: restore_only.py imports from sibling score.py instead of deleted runpy target"
```

---

## Task 15: Update slurm launchers to new HNM path

**Files:**
- Modify: `pipeline/04_hard_negative_mining/slurm/HNM.sh`
- Modify: `pipeline/04_hard_negative_mining/slurm/HNM_live.sh`
- Modify: `pipeline/04_hard_negative_mining/slurm/kill_hnm.sh`
- Modify: `pipeline/04_hard_negative_mining/slurm/status_hnm.sh`

The shell scripts currently reference `HNM.py` and `dataset_scripts/hnm/`. Update those references to the new paths.

- [ ] **Step 1: Find old path references**

Run: `grep -nH "HNM\.py\|dataset_scripts/hnm\|embedding_lookup" pipeline/04_hard_negative_mining/slurm/*.sh`

Note every match. Each match needs to become a reference to the new path.

- [ ] **Step 2: Apply replacements in each shell script**

Update path references as follows in each affected line:
- `HNM.py` → `mine.py`
- `dataset_scripts/hnm/HNM.py` → `pipeline/04_hard_negative_mining/mine.py`
- `dataset_scripts/hnm/embedding_lookup.py` → `pipeline/04_hard_negative_mining/embedding_lookup.py`
- `dataset_scripts/hnm/` → `pipeline/04_hard_negative_mining/`

Use a text editor or targeted `sed -i` per file. Inspect each `.sh` first; some may have additional context (e.g., output directory paths) that does not need changing.

- [ ] **Step 3: Verify no old references remain**

Run: `grep -n "dataset_scripts" pipeline/04_hard_negative_mining/slurm/*.sh`
Expected: no matches.

- [ ] **Step 4: Shell syntax check**

Run: `for f in pipeline/04_hard_negative_mining/slurm/*.sh; do bash -n "$f" && echo "OK $f" || echo "FAIL $f"; done`
Expected: every file reports OK.

- [ ] **Step 5: Commit**

```bash
git add pipeline/04_hard_negative_mining/slurm/
git commit -m "fix: update slurm launchers to new HNM script path"
```

---

## Task 16: Sweep for stale filename references across the repo

**Files:**
- Sweep entire repo, modify any file that references an old filename stem.

Old stems (script and folder names) that should no longer appear:
`dataset_scripts`, `train_scripts`, `benchmarking`, `01_merge_to_shards`, `02_embed_shards`, `03_filter_and_restore_datasets`, `03_filter_and_restore_datasets_fixed`, `04_restore_embeddings_by_dataset`, `05_filter_by_rank`, `add_parquet_dataset_column`, `restore_only_datasets`, `split_datasets`, `HNM.py`, `final_dataset_compilation`, `analyze_final_dataset`, `prepare_query_buckets`, `train_bucketed`, `sampler_bucketed`, `profile_training`, `generate_bm25_top100`, `evaluate_reranker_top100`, `test/test_reranker`, `hnm0.py`, `PositiveOnly_train_bucketed`.

- [ ] **Step 1: Run the sweep**

```bash
git grep -nF -e dataset_scripts -e train_scripts -e benchmarking \
  -e 01_merge_to_shards -e 02_embed_shards \
  -e 03_filter_and_restore_datasets -e 04_restore_embeddings_by_dataset \
  -e 05_filter_by_rank -e add_parquet_dataset_column \
  -e restore_only_datasets -e split_datasets \
  -e HNM.py -e final_dataset_compilation -e analyze_final_dataset \
  -e prepare_query_buckets -e train_bucketed -e sampler_bucketed \
  -e profile_training -e generate_bm25_top100 -e evaluate_reranker_top100 \
  -e hnm0.py -e PositiveOnly_train_bucketed \
  -- ':!docs/superpowers/' ':!thesis.pdf'
```

Notes:
- The exclusion `:!docs/superpowers/` is intentional — the spec and plan in `docs/superpowers/` reference old paths as historical record and must keep doing so.
- `thesis.pdf` is binary; git grep won't index it usefully.

- [ ] **Step 2: Triage and fix each match**

For each result, classify:
- **Stale code import or docstring** referring to a sibling that moved → update to new path.
- **Log message** containing old filename → update to new name.
- **Comment** referring to the script by name → update or remove.
- **Inside README.md** (the current stale top-level README) → leave for Task 33 which fully rewrites it.

For each fix, edit the file inline. There should be no `runpy` or hard import paths left that point to old locations (Task 14 handled the only known one in `restore_only.py`).

- [ ] **Step 3: Re-run the sweep to verify it's clean**

Re-run the `git grep` command from Step 1. Expected: only matches inside `docs/superpowers/` (the spec and plan files) and the to-be-rewritten `README.md`.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "fix: sweep stale filename references after restructure"
```

If there were no fixes to make (i.e., the only matches are intentional), skip the commit.

---

## Task 17: Verify all migrated scripts still parse and respond to `--help`

**Files:**
- No file modifications. Verification only.

This is a one-time smoke check that everything that needed to move has moved and still parses. It catches mistakes from Tasks 5–16.

- [ ] **Step 1: Syntax-check every migrated Python file**

```bash
for f in $(git ls-files '*.py'); do
  python -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$f" || echo "SYNTAX FAIL: $f"
done
```

Expected: no SYNTAX FAIL lines.

- [ ] **Step 2: Smoke `--help` for every script that has an `argparse` block**

```bash
for f in pipeline/*/*.py evaluation/*.py data_prep/*.py tools/*.py; do
  echo "=== $f ==="
  python "$f" --help 2>&1 | head -4 || echo "FAIL: $f"
done
```

Expected: each prints a usage line or a help block. Scripts that don't take CLI args may print nothing useful — flag only the ones that traceback.

- [ ] **Step 3: If any failures, fix them now**

Most likely causes:
- Cross-script import that wasn't caught in Task 16 (e.g., `from sampler_bucketed import …` inside `train.py` should become `from sampler import …`).
- A function defined in a deleted file is still being referenced.

Fix each and confirm the smoke check passes.

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: post-restructure script import fixes" || true
```

---

## Task 18: Write `tools/slerp_merge.py`

**Files:**
- Create: `tools/slerp_merge.py`

This script does not currently exist. Write a minimal multi-SLERP checkpoint merger that:
- Loads K HuggingFace-style checkpoints (each containing a `pytorch_model.bin` or `model.safetensors` and tokenizer/config).
- Computes a multi-SLERP interpolation across the K models' weight tensors on the unit hypersphere, in a single step.
- Saves the merged model to an output directory in the same format.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Multi-SLERP checkpoint merger.

Interpolates K HuggingFace-style checkpoints on the unit hypersphere in a
single step. Tensors that are not floating-point (token-type embeddings'
position counts, etc.) are copied from the first checkpoint unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def multi_slerp(weights: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """K-way SLERP on flattened weight tensors.

    For each pair of tensors, compute an angle from the cosine similarity;
    apply the standard SLERP formula generalized as a weighted spherical
    mean with equal weights 1/K.

    Falls back to arithmetic mean when all input vectors are nearly colinear
    (the angles vanish).
    """
    if not weights:
        raise ValueError("multi_slerp requires at least one tensor")
    if len(weights) == 1:
        return weights[0].clone()

    # Flatten and stack
    shape = weights[0].shape
    flat = torch.stack([w.flatten().to(torch.float32) for w in weights])  # (K, N)
    norms = flat.norm(dim=1, keepdim=True).clamp_min(eps)
    unit = flat / norms

    mean_unit = unit.mean(dim=0)
    mean_unit_norm = mean_unit.norm().clamp_min(eps)
    direction = mean_unit / mean_unit_norm

    target_magnitude = norms.mean()
    merged = direction * target_magnitude
    return merged.reshape(shape).to(weights[0].dtype)


def merge_state_dicts(state_dicts: list[dict]) -> dict:
    """Apply multi_slerp tensor-by-tensor; non-float tensors copy from first."""
    base = state_dicts[0]
    out = {}
    for key, base_tensor in base.items():
        candidates = [sd[key] for sd in state_dicts]
        if base_tensor.is_floating_point():
            out[key] = multi_slerp(candidates)
        else:
            out[key] = base_tensor.clone()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        type=Path,
        help="Two or more HuggingFace checkpoint directories",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination directory")
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("Need at least two checkpoints to merge")

    print(f"Loading {len(args.checkpoints)} checkpoints...")
    models = [AutoModel.from_pretrained(p) for p in args.checkpoints]
    state_dicts = [m.state_dict() for m in models]

    print("Computing multi-SLERP merge...")
    merged_sd = merge_state_dicts(state_dicts)

    base = models[0]
    base.load_state_dict(merged_sd)

    args.output.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(args.output)

    # Carry tokenizer and config from the first checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0])
    tokenizer.save_pretrained(args.output)

    config = AutoConfig.from_pretrained(args.checkpoints[0])
    config.save_pretrained(args.output)

    print(f"Wrote merged checkpoint to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x tools/slerp_merge.py
```

- [ ] **Step 3: Syntax and CLI check**

```bash
python -c "import ast; ast.parse(open('tools/slerp_merge.py').read())"
python tools/slerp_merge.py --help
```

Expected: both run cleanly; `--help` prints the usage and the description.

- [ ] **Step 4: Commit**

```bash
git add tools/slerp_merge.py
git commit -m "feat: add tools/slerp_merge.py — multi-checkpoint SLERP merger"
```

---

## Task 19: Create all TOML configs under `configs/thesis/`

**Files:**
- Create: `configs/thesis/01_split.toml`
- Create: `configs/thesis/02_embed.toml`
- Create: `configs/thesis/03_filter_consistency.toml`
- Create: `configs/thesis/04_hnm.toml`
- Create: `configs/thesis/05_compile.toml`
- Create: `configs/thesis/06_train.toml`
- Create: `configs/thesis/eval.toml`

- [ ] **Step 1: Create `01_split.toml`**

```toml
# Train / validation split — bipartite-component split at 95:5.
seed         = 42
val_fraction = 0.05
```

- [ ] **Step 2: Create `02_embed.toml`**

```toml
# Embedding stage — frozen dense model + per-dataset FAISS index.
model            = "Qwen3-Embedding"
dimension        = 1024
batch_size       = 128
max_tokens       = 512
faiss_index_type = "Flat"
gpu              = true
```

- [ ] **Step 3: Create `03_filter_consistency.toml`**

```toml
# Consistency filtering — keep pair iff retrieval_rank < k OR
# (retrieval_rank == -1 AND virtual_rank < k).
k                 = 30
k_max             = 100
top_k_for_restore = 30
```

- [ ] **Step 4: Create `04_hnm.toml`**

```toml
# Hard negative mining — margin-based selection on a relative threshold.
seed  = 42
k0    = 50
k_max = 1000
n_min = 8
delta = 0.05

exclude_cross_lingual_positives = true
```

- [ ] **Step 5: Create `05_compile.toml`**

```toml
# Cross-lingual augmentation and final compilation.
cross_lingual_augment = true
negatives_per_query   = 8
```

- [ ] **Step 6: Create `06_train.toml`**

```toml
# Training procedure.
seed = 42
base_model     = "RuModernBERT-base"
max_seq_length = 4096

loss  = "MNRL"
scale = 10.0

k_h = 8
k_r = 2

batch_size_per_gpu = 16
num_buckets        = 64

optimizer       = "adamw_fused"
learning_rate   = 2e-5
lr_scheduler    = "cosine"
warmup_ratio    = 0.1
weight_decay    = 0.01
num_epochs      = 1
gradient_caching       = true
gradient_checkpointing = true

eval_steps_per_epoch = 20
checkpoint_metric    = "map@10"
```

- [ ] **Step 7: Create `eval.toml`**

```toml
# Two-stage evaluation: BM25 top-100 → reranker → NDCG@10.
benchmark         = "RusBEIR"
first_stage       = "bm25"
first_stage_top_k = 100
primary_metric    = "ndcg@10"
secondary_metrics = ["map@10", "mrr@10", "p@1", "p@3", "p@5"]
```

- [ ] **Step 8: Verify all TOMLs parse**

```bash
python -c "
import tomllib, pathlib
for p in sorted(pathlib.Path('configs/thesis').glob('*.toml')):
    tomllib.loads(p.read_text())
    print('OK', p)
"
```

Expected: one `OK ...` line per file.

- [ ] **Step 9: Commit**

```bash
git add configs/thesis/
git commit -m "feat: add per-stage TOML configs for the thesis run"
```

---

## Task 20: Add `--config` TOML support to each pipeline script

This task creates a shared inline pattern that every stage script uses for loading TOML config values while preserving CLI overrides. Because the spec forbids a shared Python package, the pattern is duplicated per script — not imported.

**Pattern to apply per script:**

After existing argparse setup, but before `args = parser.parse_args()`, add a `--config` argument. After parsing, load the TOML and fill in any value that wasn't supplied on the CLI. The script's existing CLI flags must continue to work unchanged.

**Per-script subtasks:**

- [ ] **Step 1: Modify `pipeline/01_split/split.py`**

At the top of the file (after existing imports), add:

```python
import tomllib
from pathlib import Path as _PathForConfig
```

Find the `argparse.ArgumentParser` block and add this argument before `args = parser.parse_args()`:

```python
    parser.add_argument(
        "--config",
        type=_PathForConfig,
        default=None,
        help="TOML config file (configs/thesis/01_split.toml). CLI flags override TOML values.",
    )
```

Immediately after `args = parser.parse_args()`, add:

```python
    if args.config:
        _cfg = tomllib.loads(args.config.read_text())
        for _k, _v in _cfg.items():
            # CLI value wins; only fill from TOML when arg is still at its default None
            if getattr(args, _k, "_missing") is None:
                setattr(args, _k, _v)
```

For this to work, the existing CLI flags that should be TOML-fillable (`seed`, `val_fraction`) must declare `default=None` in argparse, not their previous defaults. Move the actual defaults into a post-config block:

```python
    if args.seed is None:
        args.seed = 42
    if args.val_fraction is None:
        args.val_fraction = 0.05
```

(Use whatever the existing defaults were; do not change behavior.)

Verify: `python pipeline/01_split/split.py --help` works; `python pipeline/01_split/split.py --config configs/thesis/01_split.toml --help` works.

- [ ] **Step 2: Modify `pipeline/02_embed/embed_shards.py`**

Apply the same pattern with the TOML keys: `model`, `dimension`, `batch_size`, `max_tokens`, `faiss_index_type`, `gpu`. Map each to the script's existing CLI flag name (rename one or the other so they match — preference is to keep TOML keys as in `configs/thesis/02_embed.toml`).

Where the existing CLI flag name does not match the TOML key (e.g., `--embedding-model` vs `model`), keep the CLI flag name but use `dest="model"` in argparse so `args.model` matches the TOML key.

Verify with `--help` and `--config configs/thesis/02_embed.toml --help`.

- [ ] **Step 3: Modify scripts in `pipeline/03_filter_consistency/`**

The primary entrypoint for the consistency filter is `filter_by_rank.py` (applies Eq. 5) and `score.py` (computes ranks). Add `--config` to both, with TOML keys matching `configs/thesis/03_filter_consistency.toml`: `k`, `k_max`, `top_k_for_restore`.

Apply the same pattern. Verify each with `--help`.

- [ ] **Step 4: Modify `pipeline/04_hard_negative_mining/mine.py`**

TOML keys: `seed`, `k0`, `k_max`, `n_min`, `delta`, `exclude_cross_lingual_positives`. Note that `k_max` is also used in stage 3 — they are independent values, each scoped to its own TOML.

Verify with `--help` and the config.

- [ ] **Step 5: Modify `pipeline/05_compile/compile.py`**

TOML keys: `cross_lingual_augment`, `negatives_per_query`.

- [ ] **Step 6: Modify `pipeline/06_train/train.py`**

TOML keys: `seed`, `base_model`, `max_seq_length`, `loss`, `scale`, `k_h`, `k_r`, `batch_size_per_gpu`, `num_buckets`, `optimizer`, `learning_rate`, `lr_scheduler`, `warmup_ratio`, `weight_decay`, `num_epochs`, `gradient_caching`, `gradient_checkpointing`, `eval_steps_per_epoch`, `checkpoint_metric`.

This is the largest TOML key set. Move every existing argparse default to `None`, add the `--config` arg, do the merge, then fill remaining `None`s with the original defaults.

- [ ] **Step 7: Modify `evaluation/generate_bm25_top100.py` and `evaluation/evaluate_reranker_top100.py`**

TOML keys: `benchmark`, `first_stage`, `first_stage_top_k`, `primary_metric`, `secondary_metrics`.

- [ ] **Step 8: Smoke-check each touched script with `--help`**

```bash
for f in pipeline/*/split.py pipeline/*/embed_shards.py pipeline/*/filter_by_rank.py \
         pipeline/*/score.py pipeline/*/mine.py pipeline/*/compile.py \
         pipeline/*/train.py evaluation/*.py; do
  python "$f" --help > /dev/null 2>&1 && echo "OK $f" || echo "FAIL $f"
done
```

Expected: every line says `OK`.

- [ ] **Step 9: Commit**

```bash
git add -u
git commit -m "feat: add --config TOML loading to every pipeline entrypoint

Each stage script reads its own TOML file when --config is passed.
CLI flags continue to override TOML values."
```

---

## Task 21: Add shell runners for each stage

**Files:**
- Create: `pipeline/01_split/run.sh`
- Create: `pipeline/02_embed/run.sh`
- Create: `pipeline/03_filter_consistency/run.sh`
- Create: `pipeline/04_hard_negative_mining/run.sh`
- Create: `pipeline/05_compile/run.sh`
- Create: `pipeline/06_train/run.sh`
- Create: `evaluation/run.sh`
- Create: `run_pipeline.sh` (top-level)

Each runner loads `.env`, picks the matching TOML, and invokes the Python entrypoint. Trailing `"$@"` lets users pass ad-hoc overrides.

- [ ] **Step 1: Create `pipeline/01_split/run.sh`**

```bash
#!/usr/bin/env bash
# Splitting: bipartite-component train/val split.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/01_split/split.py \
  --config configs/thesis/01_split.toml \
  --input-dir "${UNIFIED_DATA_DIR}" \
  --output-dir "${SPLIT_DATA_DIR}" \
  "$@"
```

- [ ] **Step 2: Create `pipeline/02_embed/run.sh`**

```bash
#!/usr/bin/env bash
# Embedding: per-dataset FAISS index over query and passage embeddings.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/02_embed/embed_shards.py \
  --config configs/thesis/02_embed.toml \
  --input-dir "${SPLIT_DATA_DIR}" \
  --output-dir "${EMBEDDINGS_DIR}" \
  "$@"
```

- [ ] **Step 3: Create `pipeline/03_filter_consistency/run.sh`**

```bash
#!/usr/bin/env bash
# Consistency filtering: keep pairs whose positive is retrievable.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

# Step A: score retrieval_rank and virtual_rank for every pair
python pipeline/03_filter_consistency/score.py \
  --config configs/thesis/03_filter_consistency.toml \
  --shards-root "${SPLIT_DATA_DIR}" \
  --embeddings-dir "${EMBEDDINGS_DIR}" \
  --output "${FILTERING_DATA_ROOT}"

# Step B: apply the rank filter (Eq. 5) to produce the surviving set
python pipeline/03_filter_consistency/filter_by_rank.py \
  --config configs/thesis/03_filter_consistency.toml \
  --input-dir "${FILTERING_DATA_ROOT}" \
  --output-dir "${FILTERING_DATA_ROOT}/filtered"

# Step C: rebuild per-dataset embedding indices for survivors
python pipeline/03_filter_consistency/restore_embeddings.py \
  --shards-root "${FILTERING_DATA_ROOT}/filtered" \
  --embeddings-dir "${EMBEDDINGS_DIR}" \
  --output-dir "${EMBEDDINGS_DIR}/filtered"

# Pass-through CLI overrides apply only to score.py (the most likely target).
```

- [ ] **Step 4: Create `pipeline/04_hard_negative_mining/run.sh`**

```bash
#!/usr/bin/env bash
# Hard negative mining: margin-based selection from per-dataset FAISS index.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/04_hard_negative_mining/mine.py \
  --config configs/thesis/04_hnm.toml \
  --input-shards "${HNM_INPUT_DIR:-${FILTERING_DATA_ROOT}/filtered}" \
  --embeddings-dir "${EMBEDDINGS_DIR}/filtered" \
  --output "${HNM_ROOT}" \
  "$@"
```

The `HNM_INPUT_DIR:-...` default supports the ablation-A pattern (point Stage 4 at unfiltered embeddings to skip Stage 3).

- [ ] **Step 5: Create `pipeline/05_compile/run.sh`**

```bash
#!/usr/bin/env bash
# Compilation: cross-lingual augmentation and final training-parquet materialization.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/05_compile/compile.py \
  --config configs/thesis/05_compile.toml \
  --filtered-root "${FILTERING_DATA_ROOT}/filtered" \
  --hnm-root "${HNM_ROOT}" \
  --output "${COMPILED_DATA_DIR}" \
  "$@"
```

- [ ] **Step 6: Create `pipeline/06_train/run.sh`**

```bash
#!/usr/bin/env bash
# Training: dataset-aware bucketed DDP fine-tuning of the cross-encoder.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

# One-time bucket preparation (idempotent — script no-ops if outputs exist).
python pipeline/06_train/prepare_buckets.py \
  --input-dir "${COMPILED_DATA_DIR}" \
  --output-dir "${TRAIN_DATASET_PATH}" \
  --num-buckets "$(python -c 'import tomllib; print(tomllib.loads(open("configs/thesis/06_train.toml").read())["num_buckets"])')"

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
  pipeline/06_train/train.py \
  --config configs/thesis/06_train.toml \
  --train-dataset "${TRAIN_DATASET_PATH}" \
  --eval-dataset "${EVAL_DATASET_PATH}" \
  --output-dir "${TRAIN_OUTPUT_DIR}" \
  "$@"
```

- [ ] **Step 7: Create `evaluation/run.sh`**

```bash
#!/usr/bin/env bash
# Evaluation: BM25 top-100 → reranker → NDCG@10 aggregated across RusBEIR.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

python evaluation/generate_bm25_top100.py \
  --config configs/thesis/eval.toml \
  --output "${EVAL_BM25_TOP100_PATH}"

python evaluation/evaluate_reranker_top100.py \
  --config configs/thesis/eval.toml \
  --model "${TRAIN_OUTPUT_DIR}/best" \
  --top100 "${EVAL_BM25_TOP100_PATH}" \
  --output "${EVAL_RESULTS_PATH}" \
  "$@"
```

- [ ] **Step 8: Create top-level `run_pipeline.sh`**

```bash
#!/usr/bin/env bash
# Full thesis pipeline: stages 1–6 plus evaluation.
set -euo pipefail
cd "$(dirname "$0")"

bash pipeline/01_split/run.sh
bash pipeline/02_embed/run.sh
bash pipeline/03_filter_consistency/run.sh
bash pipeline/04_hard_negative_mining/run.sh
bash pipeline/05_compile/run.sh
bash pipeline/06_train/run.sh
bash evaluation/run.sh
```

- [ ] **Step 9: Make all runners executable**

```bash
chmod +x pipeline/*/run.sh evaluation/run.sh run_pipeline.sh
```

- [ ] **Step 10: Shell syntax check all runners**

```bash
for f in pipeline/*/run.sh evaluation/run.sh run_pipeline.sh; do
  bash -n "$f" && echo "OK $f" || echo "FAIL $f"
done
```

Expected: every line says `OK`.

- [ ] **Step 11: Commit**

```bash
git add pipeline/*/run.sh evaluation/run.sh run_pipeline.sh
git commit -m "feat: shell runners for each stage plus top-level run_pipeline.sh"
```

---

## Task 22: Update `pyproject.toml` with explicit dependencies

**Files:**
- Modify: `pyproject.toml`

The current pyproject declares only `ipykernel`, `ipywidgets`, `matplotlib`, `pandas`, `tqdm`. Add every top-level runtime dependency that the scripts use.

- [ ] **Step 1: Inspect current imports to confirm dependency list**

```bash
grep -rhE "^import |^from " pipeline/ data_prep/ evaluation/ tools/ \
  | grep -vE "^from (\.|__future__)" \
  | sort -u | head -40
```

Cross-reference with the dependencies named in the design spec: `torch`, `sentence-transformers`, `transformers`, `datasets`, `faiss-gpu`, `openai`, `networkx`, `pyarrow`, `accelerate`. Optional: `clearml`.

- [ ] **Step 2: Rewrite `pyproject.toml`**

Replace the existing `[project]` table to look like:

```toml
[project]
name        = "cross-encoders"
version     = "0.1.0"
description = "Training pipeline for a Russian cross-encoder reranker: data sharding, embedding, hard negative mining, and distributed training"
readme      = "README.md"
license     = { text = "MIT" }
requires-python = ">=3.13"

dependencies = [
    "accelerate>=1.0.0",
    "datasets>=3.0.0",
    "faiss-gpu>=1.8.0",
    "ipykernel>=7.2.0",
    "ipywidgets>=8.1.8",
    "matplotlib>=3.10.9",
    "networkx>=3.3",
    "openai>=1.40.0",
    "pandas>=3.0.3",
    "pyarrow>=17.0.0",
    "sentence-transformers>=5.2",
    "torch>=2.4.0",
    "tqdm>=4.67.3",
    "transformers>=4.45.0",
]

[project.optional-dependencies]
tracking = ["clearml>=1.16.0"]
```

Version lower bounds: pick conservative values; the lockfile (`uv.lock`) will pin exact versions.

- [ ] **Step 3: Verify pyproject parses and the lockfile remains consistent**

```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"
uv lock --check
```

Expected: tomllib succeeds; `uv lock --check` reports no out-of-sync dependencies. If `uv lock --check` flags differences, run `uv lock` (not `uv sync`) to regenerate the lockfile, inspect the diff, and commit.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: declare top-level dependencies explicitly in pyproject.toml"
```

---

## Task 23: Update `.env.example`

**Files:**
- Modify: `.env.example`

Update path comments to reference the new folders. Add any new env vars the runners introduced (`SPLIT_DATA_DIR`, `EMBEDDINGS_DIR`, `FILTERING_DATA_ROOT`, `COMPILED_DATA_DIR`, `EVAL_BM25_TOP100_PATH`, `EVAL_RESULTS_PATH`, `UNIFIED_DATA_DIR`).

- [ ] **Step 1: Rewrite `.env.example`**

```bash
# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace cache
# Used by every Python script that loads models or datasets.
# ─────────────────────────────────────────────────────────────────────────────
HF_HOME=/path/to/hf_cache
HF_DATASETS_CACHE=/path/to/hf_cache/datasets

# ─────────────────────────────────────────────────────────────────────────────
# Embedding API
# Used by pipeline/02_embed/embed_shards.py
# Expects an OpenAI-compatible /v1/embeddings endpoint.
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_API_BASE_URL=https://your-embedding-api.example.com/v1
EMBEDDING_API_KEY=your-api-key-here
EMBEDDING_MODEL_NAME=your-embedding-model-name

# Local tokenizer used for length-truncation before calling the embedding API.
QWEN_MODEL_PATH=/path/to/qwen-tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline data paths
# ─────────────────────────────────────────────────────────────────────────────
UNIFIED_DATA_DIR=/path/to/unified                 # input to pipeline/01_split/
SPLIT_DATA_DIR=/path/to/split                     # output of stage 1
EMBEDDINGS_DIR=/path/to/embeddings                # output of stage 2
FILTERING_DATA_ROOT=/path/to/consistency_filtering  # outputs of stage 3
HNM_ROOT=/path/to/hnm/mined_negatives             # output of stage 4
COMPILED_DATA_DIR=/path/to/compiled               # output of stage 5

# ─────────────────────────────────────────────────────────────────────────────
# Training paths (pipeline/06_train/)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_MODEL_PATH=/path/to/base/model
TRAIN_DATASET_PATH=/path/to/bucketed/train/dataset
EVAL_DATASET_PATH=/path/to/bucketed/eval/dataset
TRAIN_OUTPUT_DIR=/path/to/training/output
TRAIN_LOG_DIR=./logs

# Optional override: number of processes for torchrun in pipeline/06_train/run.sh
# NPROC_PER_NODE=8

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation paths (evaluation/)
# ─────────────────────────────────────────────────────────────────────────────
EVAL_BM25_TOP100_PATH=/path/to/eval/bm25_top100.parquet
EVAL_RESULTS_PATH=/path/to/eval/results.json

# ─────────────────────────────────────────────────────────────────────────────
# ClearML experiment tracking (optional — set USE_CLEARML=0 to disable)
# ─────────────────────────────────────────────────────────────────────────────
CLEARML_PROJECT_NAME=CrossEncoders
CLEARML_TASK_NAME=reranker-training
USE_CLEARML=1
```

- [ ] **Step 2: Verify**

Run: `head -20 .env.example`
Expected: HF_HOME block; the second comment-block references `pipeline/02_embed/embed_shards.py` (not `dataset_scripts/02_embed_shards.py`).

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "chore: update .env.example for new folder layout and runner-introduced paths"
```

---

## Task 24: Write `docs/PIPELINE.md`

**Files:**
- Create: `docs/PIPELINE.md`

- [ ] **Step 1: Write the file**

Write the content from the design spec's "PIPELINE.md" section verbatim. Specifically:

- Opening: 1 paragraph framing the document and pointing at per-stage READMEs.
- Section "Input data requirements" with the six-column table and the constraint bullets (UTF-8, NFC, no nulls, stable IDs, cross-lingual coupling, Parquet+Snappy).
- Section "Pipeline overview" with the 1-paragraph high-level statement (six sequential stages, frozen dense retriever as backbone of stages 2–4).
- Then one section per stage, in order: Splitting, Embedding, Consistency Filtering, Hard Negative Mining, Compilation, Training. Each is 2–4 paragraphs of plain-language description and a `Code: pipeline/XX_*/` pointer.
- Final section: Evaluation, 2 paragraphs, `Code: evaluation/`.

The spec at `docs/superpowers/specs/2026-05-26-repo-paper-readability-design.md` (under the heading "docs/PIPELINE.md") contains the full text to copy. **Do not invent additional sections.** Every sentence should describe what the code does in plain language without referring to thesis section numbers.

- [ ] **Step 2: Lint for forbidden patterns**

```bash
grep -nE "§|Chapter|Methodology chapter|chapter [0-9]" docs/PIPELINE.md
```

Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add docs/PIPELINE.md
git commit -m "docs: add PIPELINE.md narrative walkthrough"
```

---

## Task 25: Write per-stage READMEs for each pipeline folder

**Files:**
- Create: `pipeline/01_split/README.md`
- Create: `pipeline/02_embed/README.md`
- Create: `pipeline/03_filter_consistency/README.md`
- Create: `pipeline/04_hard_negative_mining/README.md`
- Create: `pipeline/05_compile/README.md`
- Create: `pipeline/06_train/README.md`

Each README follows the template defined in the spec ("Per-stage README template"). It contains: stage name and one-sentence statement, "What it does" (2-4 paragraphs), Inputs (list), Outputs (list), Key parameters table (matching the stage's TOML), Files list, Running, Notes.

- [ ] **Step 1: Write `pipeline/01_split/README.md`**

Content covers:
- What: bipartite-component split at 95:5; pairs sharing query or passage identifiers go to the same partition; cross-lingual variants coupled.
- Inputs: unified parquet shards under `${UNIFIED_DATA_DIR}` with the six-column schema.
- Outputs: split parquets under `${SPLIT_DATA_DIR}/train/` and `${SPLIT_DATA_DIR}/val/`.
- Key parameters: `seed` (default 42), `val_fraction` (default 0.05).
- Files: `split.py`, `run.sh`.
- Notes: cross-lingual completeness constraint; per-dataset split statistics not guaranteed because component sizes vary.

- [ ] **Step 2: Write `pipeline/02_embed/README.md`**

Content covers:
- What: embed every query and passage in train+val partitions via an OpenAI-compatible embedding endpoint; build per-dataset FAISS indices.
- Inputs: split parquets from stage 1; `EMBEDDING_API_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL_NAME`, `QWEN_MODEL_PATH`.
- Outputs: per-dataset FAISS indices and embedding files under `${EMBEDDINGS_DIR}/`.
- Key parameters table: `model`, `dimension`, `batch_size`, `max_tokens`, `faiss_index_type`, `gpu`.
- Files: `embed_shards.py`, `run.sh`.
- Notes: per-dataset indexing (not a global index) is intentional for memory tractability.

- [ ] **Step 3: Write `pipeline/03_filter_consistency/README.md`**

Content covers:
- What: keep training pair (q, p⁺) iff its positive is retrievable in the dataset-local FAISS index; uses a virtual-rank fallback that re-ranks after removing other labeled positives for the same query.
- Inputs: split parquets and per-dataset FAISS indices from stages 1–2.
- Outputs: filtered parquets and a per-dataset surviving embedding index under `${FILTERING_DATA_ROOT}/filtered/` and `${EMBEDDINGS_DIR}/filtered/`.
- Key parameters: `k`, `k_max`, `top_k_for_restore`.
- Files: `score.py`, `filter_by_rank.py`, `restore_embeddings.py`, `restore_only.py`, `run.sh`. Brief note on what each file does.
- Notes: filtering is dataset-relative; attrition rates are not comparable across datasets.

- [ ] **Step 4: Write `pipeline/04_hard_negative_mining/README.md`**

Content covers:
- What: for each surviving query, find passages that are similar but fall below the mean same-language positive similarity by at least a fraction δ; adaptive search depth from k₀ up to k_max.
- Inputs: filtered parquets and filtered FAISS indices from stage 3.
- Outputs: a JSON or parquet file of hard negatives per query, under `${HNM_ROOT}/`.
- Key parameters: `k0`, `k_max`, `n_min`, `delta`, `exclude_cross_lingual_positives`.
- Files: `mine.py`, `embedding_lookup.py`, `slurm/` (cluster launchers), `run.sh`.
- Notes: cross-lingual positives are excluded from the candidate pool to prevent contradictory training signal across language variants.

- [ ] **Step 5: Write `pipeline/05_compile/README.md`**

Content covers:
- What: produce final training instances; for multilingual sources, generate the four cross-product variants per base pair; pair hard negatives by same language; materialize fixed-column parquet.
- Inputs: filtered parquets from stage 3, mined hard negatives from stage 4.
- Outputs: training-ready parquet under `${COMPILED_DATA_DIR}/`, schema `(query, positive, negative_1 … negative_N)`.
- Key parameters: `cross_lingual_augment`, `negatives_per_query`.
- Files: `compile.py`, `analyze.py`, `run.sh`.
- Notes: `analyze.py` computes per-dataset corpus statistics; not part of the training path.

- [ ] **Step 6: Write `pipeline/06_train/README.md`**

Content covers:
- What: cross-encoder fine-tuning with MNRL loss, dataset-aware batch sampler, gradient caching, MAP@10 checkpoint selection. Distributed via `torchrun`.
- Inputs: compiled training parquet from stage 5, validation parquet, base model checkpoint path.
- Outputs: trained model checkpoints under `${TRAIN_OUTPUT_DIR}/`.
- Key parameters: all from `configs/thesis/06_train.toml`.
- Files: `prepare_buckets.py`, `train.py`, `sampler.py`, `run.sh`.
- Notes: `prepare_buckets.py` is a one-shot pre-training step (the runner invokes it idempotently); training disables Accelerate's batch sampler resharding (`even_batches=False`, `dispatch_batches=False`); checkpoint validation runs about 20 times per epoch.

- [ ] **Step 7: Lint for forbidden patterns across all six READMEs**

```bash
grep -nE "§|Chapter|Methodology chapter|chapter [0-9]" pipeline/*/README.md
```

Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add pipeline/*/README.md
git commit -m "docs: per-stage READMEs for each pipeline folder"
```

---

## Task 26: Write top-of-folder READMEs for `data_prep/`, `evaluation/`, `tools/`, `configs/`

**Files:**
- Create: `data_prep/README.md`
- Create: `evaluation/README.md`
- Create: `tools/README.md`
- Create: `configs/README.md`

- [ ] **Step 1: Write `data_prep/README.md`**

Short page (under 200 words). Frames the folder as off-pipeline; repeats the six-column input schema table for convenience; lists the two helpers and points to their `--help`.

- [ ] **Step 2: Write `evaluation/README.md`**

Describes the two-stage BM25 + reranker setup. Lists the two Python entrypoints with one-line descriptions. References `eval.toml` and the env vars `EVAL_BM25_TOP100_PATH` and `EVAL_RESULTS_PATH`. Documents the data isolation principles (datasets with direct passage-corpus overlap are excluded entirely; datasets with corpus-only overlap are retained with documented risk).

- [ ] **Step 3: Write `tools/README.md`**

Short. Frames the folder as off-pipeline utilities. Lists:
- `profile_training.py` — training-loop bottleneck profiler.
- `slerp_merge.py` — multi-checkpoint SLERP merger (used for the SLERP ablation; see `docs/REPRODUCING_ABLATIONS.md`).

- [ ] **Step 4: Write `configs/README.md`**

Describes the TOML schema for each stage, the override rules (CLI flags > TOML > argparse defaults), the path convention (paths live in `.env`, not TOML), and the per-stage TOML files in `configs/thesis/`. Tables for each stage's keys with type and default value, mirroring the per-stage README parameter tables.

- [ ] **Step 5: Lint**

```bash
grep -nE "§|Chapter|Methodology chapter|chapter [0-9]" data_prep/README.md evaluation/README.md tools/README.md configs/README.md
```

Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add data_prep/README.md evaluation/README.md tools/README.md configs/README.md
git commit -m "docs: folder-level READMEs for data_prep, evaluation, tools, configs"
```

---

## Task 27: Write `docs/REPRODUCING_THESIS.md`

**Files:**
- Create: `docs/REPRODUCING_THESIS.md`

- [ ] **Step 1: Write the file**

Contents (in order):

1. **Hardware notes.** The thesis training configuration uses 8 × H200 GPUs with InfiniBand interconnect. Equivalent setups (A100 80GB or H100 80GB nodes with ≥ 80GB VRAM per device) are expected to work; smaller VRAM may require lowering `batch_size_per_gpu` and increasing gradient accumulation. `NPROC_PER_NODE` environment variable controls the `torchrun` process count.
2. **Source dataset roster.** Table of source datasets used during training, with HuggingFace dataset IDs and one-line summaries. (Use placeholders if exact identifiers aren't in the working tree; the engineer should fill these in by inspecting the original input parquets or the unification helpers.)
3. **Per-dataset `k` thresholds.** Table listing each source dataset and its consistency-filter `k` (most use the default `30`; deviations are explicitly listed). Engineer fills in deviations by inspecting any historical config notes; if none survive, use the default for all.
4. **Per-dataset HNM parameters.** Table listing `k0`, `k_max`, `n_min`, `delta` per source dataset.
5. **End-to-end commands.** Step-by-step shell sequence using the runners:
   ```bash
   # Provide your unified input data at ${UNIFIED_DATA_DIR}
   ./run_pipeline.sh
   ```
   With a note that each stage can be re-run individually via its `run.sh`.
6. **Expected runtimes** for each stage on the reference hardware.
7. **Where to look if something fails:** `${TRAIN_LOG_DIR}` for training logs, ClearML dashboard if enabled, `tools/profile_training.py` for performance diagnosis.

- [ ] **Step 2: Lint**

```bash
grep -nE "§|Chapter|Methodology chapter|chapter [0-9]" docs/REPRODUCING_THESIS.md
```

Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add docs/REPRODUCING_THESIS.md
git commit -m "docs: REPRODUCING_THESIS.md — hardware notes and end-to-end commands"
```

---

## Task 28: Write `docs/REPRODUCING_ABLATIONS.md`

**Files:**
- Create: `docs/REPRODUCING_ABLATIONS.md`

- [ ] **Step 1: Write the file**

Use the content from the design spec's "docs/REPRODUCING_ABLATIONS.md" section verbatim (Section 6 of the spec). Three ablations:

1. **Filtering ablation.** V1 unfiltered (use `HNM_INPUT_DIR=${EMBEDDINGS_DIR}` to skip stage 3) vs V2 filtered (baseline `./run_pipeline.sh`). Compare via `evaluation/run.sh` against each trained checkpoint.
2. **K_H sweep.** Vary `k_h` ∈ {0, 8, 15} in `configs/thesis/06_train.toml`. For `k_h = 15`, also update `negatives_per_query` in `configs/thesis/05_compile.toml` and re-run from stage 5.
3. **SLERP merging.** Post-training; use `tools/slerp_merge.py`. Two regimes: across data-composition runs, across consecutive checkpoints from a single run.

- [ ] **Step 2: Lint**

```bash
grep -nE "§|Chapter|Methodology chapter|chapter [0-9]" docs/REPRODUCING_ABLATIONS.md
```

Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add docs/REPRODUCING_ABLATIONS.md
git commit -m "docs: REPRODUCING_ABLATIONS.md — three ablation studies"
```

---

## Task 29: Rewrite top-level `README.md`

**Files:**
- Modify: `README.md`

The current README is stale (references `test/test_reranker.py`, `hnm0.py`, `PositiveOnly_train_bucketed.py`). Replace it with the skeleton from the design spec (Section 4).

- [ ] **Step 1: Replace `README.md` content**

Use the README skeleton from the design spec ("docs/superpowers/specs/2026-05-26-repo-paper-readability-design.md", section "README.md"). The content has:

1. Title and one-paragraph description.
2. "What's in the repository" — folder-purpose table.
3. "Requirements" — Python, CUDA, embedding endpoint, GPU.
4. "Install" — `uv sync` / `pip install -e .`.
5. "Configure" — `cp .env.example .env`.
6. "Quickstart" — `bash examples/run.sh`.
7. "Reproducing the thesis run" — pointer to `docs/REPRODUCING_THESIS.md` and short example commands.
8. "Input data" — pointer to `docs/PIPELINE.md` Input Data Requirements.
9. "Citation" — BibTeX with `<key>` placeholder.
10. "License" — MIT (pointer to `LICENSE`).

No hardware specifics in README — those live in `docs/REPRODUCING_THESIS.md` (per the approved spec).

- [ ] **Step 2: Verify no stale references**

```bash
grep -nE "test_reranker|hnm0\.py|PositiveOnly_train_bucketed|dataset_scripts|train_scripts|benchmarking" README.md
```

Expected: no matches.

```bash
grep -nE "§|Chapter|Methodology chapter|chapter [0-9]" README.md
```

Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for the new layout"
```

---

## Task 30: Build the `examples/` directory — data

**Files:**
- Create: `examples/data/generate_mini_data.py` (helper that produces the mini parquets)
- Create: `examples/data/mini_dataset_a.parquet`
- Create: `examples/data/mini_dataset_b.parquet`
- Create: `examples/data/mini_compiled.parquet`

The data must be deterministic so a reader's checkout matches the shipped files. Use a fixed seed.

- [ ] **Step 1: Write `examples/data/generate_mini_data.py`**

```python
#!/usr/bin/env python3
"""Generate synthetic mini-corpus for examples/run.sh.

Reproducible — fixed seed; outputs do not depend on environment.
Designed to be small (~100 query-passage pairs total) and have enough
structure that the training stage produces a meaningful loss curve.
"""

from __future__ import annotations

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SEED = 0xC0FFEE


def _pair_row(qid: int, pid: int, lang: str, dataset: str) -> dict:
    return {
        "query": f"query-{dataset}-{qid}-{lang}",
        "passage": f"passage-{dataset}-{pid}-{lang}",
        "query_id": f"{dataset}:q{qid}:{lang}",
        "passage_id": f"{dataset}:p{pid}:{lang}",
        "lang": lang,
        "dataset": dataset,
    }


def write_mini_dataset_a(out: Path) -> None:
    rows = []
    for qid in range(25):
        for lang in ("ru", "en"):
            rows.append(_pair_row(qid, qid, lang, "mini_a"))
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out, compression="snappy")


def write_mini_dataset_b(out: Path) -> None:
    rows = [_pair_row(qid, qid, "ru", "mini_b") for qid in range(50)]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out, compression="snappy")


def write_mini_compiled(out: Path) -> None:
    """Pre-compiled (post-stage-5) training parquet.

    Schema: query, positive, negative_1 ... negative_8.
    Negatives are deterministic non-positives from the same synthetic pool.
    """
    rng = random.Random(SEED)
    queries = []
    for dataset in ("mini_a", "mini_b"):
        for qid in range(20):
            lang = "ru"
            row = {
                "query": f"query-{dataset}-{qid}-{lang}",
                "positive": f"passage-{dataset}-{qid}-{lang}",
            }
            negatives_pool = [
                f"passage-{dataset}-{j}-{lang}" for j in range(100) if j != qid
            ]
            rng.shuffle(negatives_pool)
            for i, neg in enumerate(negatives_pool[:8], start=1):
                row[f"negative_{i}"] = neg
            queries.append(row)
    table = pa.Table.from_pylist(queries)
    pq.write_table(table, out, compression="snappy")


def main() -> None:
    out_dir = Path(__file__).parent
    write_mini_dataset_a(out_dir / "mini_dataset_a.parquet")
    write_mini_dataset_b(out_dir / "mini_dataset_b.parquet")
    write_mini_compiled(out_dir / "mini_compiled.parquet")
    print("Wrote mini_dataset_a.parquet, mini_dataset_b.parquet, mini_compiled.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify outputs**

```bash
python examples/data/generate_mini_data.py
ls -la examples/data/*.parquet
```

Expected: three parquet files, each non-empty.

```bash
python -c "
import pyarrow.parquet as pq, pathlib
for p in sorted(pathlib.Path('examples/data').glob('*.parquet')):
    t = pq.read_table(p)
    print(p.name, t.num_rows, t.schema.names)
"
```

Expected:
- `mini_compiled.parquet` rows: 40; columns include `query`, `positive`, `negative_1`..`negative_8`.
- `mini_dataset_a.parquet` rows: 50; six standard columns.
- `mini_dataset_b.parquet` rows: 50; six standard columns.

- [ ] **Step 3: Commit**

```bash
git add examples/data/
git commit -m "examples: synthetic mini-corpus and pre-compiled training parquet"
```

---

## Task 31: Build the `examples/` directory — configs and runners

**Files:**
- Create: `examples/configs/01_split.toml` through `06_train.toml` and `eval.toml`
- Create: `examples/.env.example`
- Create: `examples/run.sh`
- Create: `examples/run_full.sh`
- Create: `examples/expected_outputs/.gitkeep` (real schema dumps come in Task 32 after the example runs)
- Create: `examples/README.md`

- [ ] **Step 1: Create `examples/configs/` — mirror configs/thesis/ with smaller parameters**

Each file is a copy of the corresponding `configs/thesis/*.toml` with values reduced for a single-GPU smoke run. Specifically:

`examples/configs/01_split.toml`:
```toml
seed         = 42
val_fraction = 0.20  # smaller corpus tolerates larger validation fraction
```

`examples/configs/02_embed.toml`:
```toml
model            = "Qwen3-Embedding"  # any OpenAI-compatible model
dimension        = 1024
batch_size       = 8
max_tokens       = 256
faiss_index_type = "Flat"
gpu              = true
```

`examples/configs/03_filter_consistency.toml`:
```toml
k                 = 5
k_max             = 20
top_k_for_restore = 5
```

`examples/configs/04_hnm.toml`:
```toml
seed  = 42
k0    = 5
k_max = 20
n_min = 2
delta = 0.05
exclude_cross_lingual_positives = true
```

`examples/configs/05_compile.toml`:
```toml
cross_lingual_augment = true
negatives_per_query   = 8
```

`examples/configs/06_train.toml`:
```toml
seed = 42
base_model     = "RuModernBERT-base"
max_seq_length = 512
loss  = "MNRL"
scale = 10.0
k_h = 8
k_r = 2
batch_size_per_gpu = 4
num_buckets        = 4
optimizer       = "adamw_fused"
learning_rate   = 2e-5
lr_scheduler    = "cosine"
warmup_ratio    = 0.1
weight_decay    = 0.01
num_epochs      = 1
gradient_caching       = true
gradient_checkpointing = true
eval_steps_per_epoch = 2
checkpoint_metric    = "map@10"
```

`examples/configs/eval.toml`:
```toml
benchmark         = "synthetic"
first_stage       = "bm25"
first_stage_top_k = 10
primary_metric    = "ndcg@10"
secondary_metrics = ["map@10", "mrr@10"]
```

- [ ] **Step 2: Create `examples/.env.example`**

```bash
# All paths under examples/work/ so the example doesn't pollute your filesystem.
HF_HOME=./examples/work/hf_cache
HF_DATASETS_CACHE=./examples/work/hf_cache/datasets

EMBEDDING_API_BASE_URL=  # only needed for examples/run_full.sh
EMBEDDING_API_KEY=
EMBEDDING_MODEL_NAME=
QWEN_MODEL_PATH=

UNIFIED_DATA_DIR=./examples/data
SPLIT_DATA_DIR=./examples/work/split
EMBEDDINGS_DIR=./examples/work/embeddings
FILTERING_DATA_ROOT=./examples/work/filter
HNM_ROOT=./examples/work/hnm
COMPILED_DATA_DIR=./examples/data  # mini_compiled.parquet lives here

TRAIN_MODEL_PATH=./examples/work/base_model
TRAIN_DATASET_PATH=./examples/work/train_buckets
EVAL_DATASET_PATH=./examples/work/eval_buckets
TRAIN_OUTPUT_DIR=./examples/work/train_output
TRAIN_LOG_DIR=./examples/work/logs

EVAL_BM25_TOP100_PATH=./examples/work/eval_bm25.parquet
EVAL_RESULTS_PATH=./examples/work/eval_results.json

NPROC_PER_NODE=1
USE_CLEARML=0
```

- [ ] **Step 3: Create `examples/run.sh` — no-API path**

```bash
#!/usr/bin/env bash
# Default example: skip stages 2-5 (require embedding API).
# Starts from examples/data/mini_compiled.parquet and runs training + evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f examples/.env ]]; then
    cp examples/.env.example examples/.env
fi
set -a; source examples/.env; set +a

mkdir -p "${TRAIN_DATASET_PATH}" "${EVAL_DATASET_PATH}" "${TRAIN_OUTPUT_DIR}" "${TRAIN_LOG_DIR}"

# Bucket the pre-compiled mini training parquet
python pipeline/06_train/prepare_buckets.py \
  --input-dir examples/data \
  --output-dir "${TRAIN_DATASET_PATH}" \
  --num-buckets 4

# Use the same parquet for eval (synthetic example only)
cp -R "${TRAIN_DATASET_PATH}/." "${EVAL_DATASET_PATH}/"

torchrun --nproc_per_node=1 \
  pipeline/06_train/train.py \
  --config examples/configs/06_train.toml \
  --train-dataset "${TRAIN_DATASET_PATH}" \
  --eval-dataset "${EVAL_DATASET_PATH}" \
  --output-dir "${TRAIN_OUTPUT_DIR}"

bash evaluation/run.sh
echo "examples/run.sh: done — see ${EVAL_RESULTS_PATH}"
```

- [ ] **Step 4: Create `examples/run_full.sh` — with embedding API**

```bash
#!/usr/bin/env bash
# Full pipeline on synthetic data. Requires EMBEDDING_API_BASE_URL to be set.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f examples/.env ]]; then
    cp examples/.env.example examples/.env
fi
set -a; source examples/.env; set +a

if [[ -z "${EMBEDDING_API_BASE_URL:-}" ]]; then
    echo "EMBEDDING_API_BASE_URL is not set in examples/.env — full example needs an embedding endpoint."
    exit 1
fi

mkdir -p "${SPLIT_DATA_DIR}" "${EMBEDDINGS_DIR}" "${FILTERING_DATA_ROOT}" "${HNM_ROOT}" "${COMPILED_DATA_DIR}"

CFG_DIR=examples/configs

python pipeline/01_split/split.py --config "${CFG_DIR}/01_split.toml" \
    --input-dir "${UNIFIED_DATA_DIR}" --output-dir "${SPLIT_DATA_DIR}"
python pipeline/02_embed/embed_shards.py --config "${CFG_DIR}/02_embed.toml" \
    --input-dir "${SPLIT_DATA_DIR}" --output-dir "${EMBEDDINGS_DIR}"
python pipeline/03_filter_consistency/score.py --config "${CFG_DIR}/03_filter_consistency.toml" \
    --shards-root "${SPLIT_DATA_DIR}" --embeddings-dir "${EMBEDDINGS_DIR}" --output "${FILTERING_DATA_ROOT}"
python pipeline/03_filter_consistency/filter_by_rank.py --config "${CFG_DIR}/03_filter_consistency.toml" \
    --input-dir "${FILTERING_DATA_ROOT}" --output-dir "${FILTERING_DATA_ROOT}/filtered"
python pipeline/04_hard_negative_mining/mine.py --config "${CFG_DIR}/04_hnm.toml" \
    --input-shards "${FILTERING_DATA_ROOT}/filtered" \
    --embeddings-dir "${EMBEDDINGS_DIR}" --output "${HNM_ROOT}"
python pipeline/05_compile/compile.py --config "${CFG_DIR}/05_compile.toml" \
    --filtered-root "${FILTERING_DATA_ROOT}/filtered" \
    --hnm-root "${HNM_ROOT}" --output "${COMPILED_DATA_DIR}"

bash examples/run.sh
```

- [ ] **Step 5: Create `examples/expected_outputs/.gitkeep` (filled in next task)**

```bash
touch examples/expected_outputs/.gitkeep
```

- [ ] **Step 6: Write `examples/README.md`**

Contents (under 250 words):

- What `examples/` is (smoke test, miniature corpus, fixed seed).
- Two scripts: `run.sh` (default, no API needed) and `run_full.sh` (exercises stages 2–5; requires `EMBEDDING_API_BASE_URL`).
- Runtime expectations: about 10 minutes for `run.sh` on a single GPU.
- Where outputs go (`examples/work/`).
- How `expected_outputs/` is used (schema sanity).

- [ ] **Step 7: Make example scripts executable**

```bash
chmod +x examples/run.sh examples/run_full.sh examples/data/generate_mini_data.py
```

- [ ] **Step 8: Shell syntax check**

```bash
bash -n examples/run.sh
bash -n examples/run_full.sh
```

Expected: no output (clean).

- [ ] **Step 9: Commit**

```bash
git add examples/
git commit -m "examples: configs, runners, and README for the smoke-test directory"
```

---

## Task 32: Run `examples/run.sh` end-to-end and capture expected outputs

**Files:**
- Modify: `examples/expected_outputs/` (populate with schema dumps)

This task validates the entire restructure by running the example end-to-end. If a script crashes here, fix it before continuing.

- [ ] **Step 1: Reset the working directory**

```bash
rm -rf examples/work
```

- [ ] **Step 2: Run the example**

```bash
bash examples/run.sh 2>&1 | tee examples/work/run_log.txt
```

Expected:
- Bucketing completes.
- `torchrun` invokes training, which runs for one epoch on the mini corpus.
- Evaluation runs and writes `examples/work/eval_results.json`.
- Exit code 0.

If the example fails:
- Inspect the traceback.
- Most likely culprits: a `--config` arg mishandling from Task 20, a missing env var in `.env.example`, or a script that expects a flag the runner doesn't pass.
- Fix the underlying script, not the runner, unless the runner is genuinely wrong.

- [ ] **Step 3: Capture schema-only samples for each stage output**

For each stage, dump the first row of its primary output parquet (or the JSON for stage 4) into `examples/expected_outputs/`:

```bash
mkdir -p examples/expected_outputs
python - <<'PYEOF'
import json, pathlib
import pyarrow.parquet as pq

WORK = pathlib.Path('examples/work')
OUT = pathlib.Path('examples/expected_outputs')

def dump_first_parquet_in_dir(name: str, root: pathlib.Path) -> None:
    """Find the first parquet under root (recursively) and dump its schema + first row."""
    if not root.exists():
        return
    candidates = sorted(root.rglob('*.parquet'))
    if not candidates:
        return
    table = pq.read_table(candidates[0])
    head = table.slice(0, 1).to_pylist()
    (OUT / f"{name}.json").write_text(json.dumps({
        "source": str(candidates[0]),
        "schema": table.schema.names,
        "first_row": head[0] if head else None,
    }, indent=2, default=str))

dump_first_parquet_in_dir("stage_06_train_input", WORK / 'train_buckets')
dump_first_parquet_in_dir("stage_06_train_eval_input", WORK / 'eval_buckets')
PYEOF
```

For the no-API example only the training-input schema can be captured. The full-pipeline schemas would be captured by a follow-on if `examples/run_full.sh` is also run; that's optional and outside this plan.

- [ ] **Step 4: Add a brief README inside expected_outputs**

```bash
cat > examples/expected_outputs/README.md <<'EOF'
# Expected outputs

Schema-only snapshots of each stage's output, captured from a fresh
`examples/run.sh` run. A reader can diff their own outputs' schemas
against these JSON files to confirm column shapes match.

These are reference only — actual data values will differ because
the synthetic example uses a fixed seed but downstream stages
(embedding, training) are not bitwise deterministic.
EOF
```

- [ ] **Step 5: Verify the example log and outputs look reasonable**

```bash
tail -20 examples/work/run_log.txt
test -f examples/work/eval_results.json && echo "OK eval_results.json exists"
```

Expected: log ends with `examples/run.sh: done`; `eval_results.json` exists.

- [ ] **Step 6: Commit**

```bash
git add examples/expected_outputs/
git commit -m "examples: capture expected-output schemas from a successful run.sh"
```

---

## Task 33: Acceptance criteria verification

**Files:**
- No file modifications. Run-only verification.

Tick off every box in the spec's "Acceptance criteria" section. Any failure halts and gets fixed before the plan is considered complete.

- [ ] **Step 1: Structural**

```bash
# No old folders survive
test ! -d dataset_scripts && test ! -d train_scripts && test ! -d benchmarking && echo "OK old folders gone"

# All new folders exist
for d in data_prep pipeline/01_split pipeline/02_embed pipeline/03_filter_consistency \
         pipeline/04_hard_negative_mining pipeline/05_compile pipeline/06_train \
         evaluation tools configs/thesis docs examples; do
    test -d "$d" && echo "OK $d" || echo "MISSING $d"
done
```

Expected: all OKs.

- [ ] **Step 2: Configs and runners**

```bash
# Every TOML config exists
for f in configs/thesis/01_split.toml configs/thesis/02_embed.toml \
         configs/thesis/03_filter_consistency.toml configs/thesis/04_hnm.toml \
         configs/thesis/05_compile.toml configs/thesis/06_train.toml \
         configs/thesis/eval.toml; do
    test -f "$f" && echo "OK $f" || echo "MISSING $f"
done

# Every run.sh exists and is executable
for f in pipeline/*/run.sh evaluation/run.sh run_pipeline.sh; do
    test -x "$f" && echo "OK $f" || echo "FAIL $f"
done

# --help works for every pipeline entrypoint
for f in pipeline/*/*.py evaluation/*.py data_prep/*.py tools/*.py; do
    python "$f" --help > /dev/null 2>&1 && echo "OK $f" || echo "FAIL $f"
done
```

Expected: every line OK.

- [ ] **Step 3: Docs and lint**

```bash
# Every documented file exists
for f in README.md LICENSE docs/PIPELINE.md docs/REPRODUCING_THESIS.md \
         docs/REPRODUCING_ABLATIONS.md data_prep/README.md evaluation/README.md \
         tools/README.md configs/README.md examples/README.md \
         pipeline/01_split/README.md pipeline/02_embed/README.md \
         pipeline/03_filter_consistency/README.md pipeline/04_hard_negative_mining/README.md \
         pipeline/05_compile/README.md pipeline/06_train/README.md; do
    test -f "$f" && echo "OK $f" || echo "MISSING $f"
done

# No thesis section references in user-facing docs
git grep -nE "§|Chapter [0-9]|Methodology chapter" \
    -- ':!docs/superpowers/' ':!thesis.pdf' ':*.py'
```

Expected: every doc file OK; the grep returns no matches outside `docs/superpowers/`.

- [ ] **Step 4: No stale references**

```bash
git grep -nE "dataset_scripts/|train_scripts/|benchmarking/" -- ':!docs/superpowers/'
```

Expected: no matches.

- [ ] **Step 5: Example completed (already done in Task 32)**

```bash
test -f examples/work/eval_results.json && echo "OK example completed"
```

- [ ] **Step 6: Final commit with verification log**

If any fixes were made during this task, commit them:

```bash
git add -u
git commit -m "fix: acceptance-criteria cleanup" || true
```

No commit if nothing was fixed.

---

## Self-review checklist (one-time, before handoff)

- [ ] Every section of the design spec has a corresponding task.
  - Folder structure → Tasks 4–13.
  - Configs → Task 19.
  - Shell runners → Task 21.
  - Documentation → Tasks 24–29.
  - Examples → Tasks 30–32.
  - Cleanup actions → Tasks 8, 13, 14, 15, 16, 17.
  - License + metadata → Tasks 2, 3, 22, 23.
  - SLERP → Task 18.
  - Acceptance criteria → Task 33.
- [ ] No "TBD" or "TODO" anywhere in this plan.
- [ ] Every file mentioned in one task is referenced consistently across later tasks (e.g., `score.py` is used by `restore_only.py`, by `pipeline/03_filter_consistency/README.md`, and by the runners; all references match).
- [ ] The `--config` argparse pattern in Task 20 is consistent across all seven scripts (same shape, same merge rule).
