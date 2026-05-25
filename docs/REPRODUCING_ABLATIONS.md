# Reproducing the ablations

Three ablation studies are documented here. Each is self-contained:
its inputs are the same `${UNIFIED_DATA_DIR}` used for the baseline
thesis run, its commands are variants of the standard runners, and
its outputs are scored against the same `evaluation/` suite. The
baseline against which every ablation is compared is the run produced
by `./run_pipeline.sh` against `configs/thesis/`.

## A. Filtering ablation — V1 unfiltered vs V2 filtered

The consistency filter in stage 3 is the most invasive single-stage
choice in the pipeline; this ablation measures its end-to-end effect by
training two checkpoints under identical hyperparameters, one with the
filter and one without.

### V2 filtered (baseline)

Standard run:

    ./run_pipeline.sh
    bash evaluation/run.sh

V2 is the baseline reranker described in the thesis. No overrides.

### V1 unfiltered

Re-run the pipeline but skip stage 3. The hard-negative miner reads
directly from the unfiltered embeddings produced by stage 2, which is
controlled by the `HNM_INPUT_DIR` environment variable:

    bash pipeline/01_split/run.sh
    bash pipeline/02_embed/run.sh
    # skip stage 3
    HNM_INPUT_DIR=${EMBEDDINGS_DIR} bash pipeline/04_hard_negative_mining/run.sh
    bash pipeline/05_compile/run.sh
    bash pipeline/06_train/run.sh

For evaluation, point `TRAIN_OUTPUT_DIR` (or the trained-model path
expected by `evaluate_reranker_top100.py`) at the V1 output and run:

    bash evaluation/run.sh

The training Parquet for V1 is necessarily larger than V2 (no rows
removed by filtering); expect a higher wall-clock at stage 6 and
slightly different MAP@10 trajectories. Compare the V1 and V2
`${EVAL_RESULTS_PATH}` files side-by-side; NDCG@10 averaged across the
benchmark suite is the headline comparison.

## B. K_H sweep

The number of hard negatives per query is a primary hyperparameter of
the contrastive loss. This ablation sweeps `k_h ∈ {0, 8, 15}` while
keeping `k_r = 2` fixed.

### k_h = 0 and k_h = 8

For `k_h = 0` (pure in-batch random negatives) and `k_h = 8`
(baseline), only the trainer needs to change. The compiled Parquet
from the baseline run already carries 8 hard negatives per row, which
is enough for both settings.

Edit `configs/thesis/06_train.toml`:

    k_h = 0    # or k_h = 8 for the baseline

Then re-run training and evaluation only:

    bash pipeline/06_train/run.sh
    bash evaluation/run.sh

Use a distinct `TRAIN_OUTPUT_DIR` per sweep point to keep the
checkpoints separable.

### k_h = 15

`k_h = 15` requires more hard negatives per row than the baseline
compilation produces. Both the compilation stage and the training
stage need an update.

Edit `configs/thesis/05_compile.toml`:

    negatives_per_query = 15

Edit `configs/thesis/06_train.toml`:

    k_h = 15

Re-run from stage 5 onwards:

    bash pipeline/05_compile/run.sh
    bash pipeline/06_train/run.sh
    bash evaluation/run.sh

Stages 1–4 are unchanged. Stage 5 will drop more queries than the
baseline (those whose mined-negative pool is shorter than 15); the
resulting training Parquet is therefore not the baseline-minus-some-rows
set, and the comparison is fairest when all three sweep points are
trained and evaluated independently.

## C. SLERP checkpoint merging

The SLERP ablation is post-training: it composes existing checkpoints
into a new one without any additional training. The merge is performed
by `tools/slerp_merge.py`, which takes a list of checkpoints and
matching weights and produces a single merged checkpoint by spherical
linear interpolation across the parameter tensors.

Two regimes are documented.

### C1. Across data-composition runs

Train several checkpoints from the same initialisation but with
different `${UNIFIED_DATA_DIR}` compositions (e.g. one excluding a
single source dataset, one with reweighted dataset frequencies). After
training, merge:

    python tools/slerp_merge.py \
      --inputs /path/to/run_a/best /path/to/run_b/best /path/to/run_c/best \
      --weights 0.34 0.33 0.33 \
      --output /path/to/merged_composition

Then evaluate the merged checkpoint:

    bash evaluation/run.sh   # with the merged checkpoint as the reranker

The intent is to combine complementary expertise from several
data-composition runs into one model without retraining on the union.

### C2. Across consecutive checkpoints of a single run

A single training run writes one checkpoint per validation step (about
twenty per epoch under the default `eval_steps_per_epoch`). The
end-of-training checkpoint is rarely the best on the validation set;
merging the last few high-MAP@10 checkpoints can outperform any single
checkpoint.

Identify the top-N checkpoints by validation MAP@10 from the training
log, then merge them:

    python tools/slerp_merge.py \
      --inputs ${TRAIN_OUTPUT_DIR}/checkpoint-1900 \
               ${TRAIN_OUTPUT_DIR}/checkpoint-2000 \
               ${TRAIN_OUTPUT_DIR}/checkpoint-2100 \
      --weights 0.33 0.34 0.33 \
      --output ${TRAIN_OUTPUT_DIR}/slerp_consecutive

Evaluate the merged checkpoint via `evaluation/run.sh`. Equal weights
are a sensible starting point; weights proportional to the validation
MAP@10 of each input is a small refinement.

Both regimes are post-hoc — they do not require re-running stages 1–6.
The only cost is one evaluation pass per merge.
