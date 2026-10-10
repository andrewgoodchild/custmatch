#!/bin/bash
# Remaining Ditto (Li et al., PVLDB 2021) datasets.
# NOTE: MixDA (--da) is OFF. With the repo's prescribed drop_col operator, Walmart-Amazon
# collapsed to predicting all-positive (17.2 F1 after 15 epochs); without it the same setup
# reached 82.6 F1 after 2. These runs therefore correspond to the paper's Ditto(DK) column.
# RoBERTa-base, lr 3e-5, max_len 128, batch 16, MixDA operator per the repo's
# run_all_er_magellan.py, --dk general, 40 epochs on the small sets and 15 otherwise.
# Ordered shortest-first so results land early; DBLP-GoogleScholar is last (~5h on its own).
set -u
cd "$(dirname "$0")/../external/ditto" || exit 1
PY=../../.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
LOGDIR=../../results/ditto
mkdir -p "$LOGDIR"

run() {  # task, da_op (unused, kept for the record), epochs
  local task=$1 op=$2 ep=$3
  local tag=${task//\//_}
  echo "=== START $task (da=$op, epochs=$ep) $(date +%T)"
  PYTHONUNBUFFERED=1 nice $PY train_ditto.py \
    --task "$task" --logdir results_ditto/ --finetuning \
    --batch_size 16 --max_len 128 --lr 3e-5 --lm roberta \
    --n_epochs "$ep" --run_id 0 --dk general \
    > "$LOGDIR/$tag.log" 2>&1
  echo "=== DONE  $task $(date +%T): $(grep -h '^epoch' "$LOGDIR/$tag.log" | tail -1)"
}

run Structured/Beer               drop_col   40
run Structured/iTunes-Amazon      drop_col   40
run Structured/Walmart-Amazon     drop_col   15
run Structured/Amazon-Google      swap       15
run Structured/DBLP-ACM           swap       15
run Structured/DBLP-GoogleScholar swap       15
echo "=== ALL DONE $(date +%T)"
