#!/bin/bash
# Reproduce Ditto (Li et al., PVLDB 2021) on Apple MPS.
# Paper settings: RoBERTa-base, lr 3e-5, max_len 256, batch 32. Here: max_len 128 and
# batch 16 - with MixDA the effective batch doubles and 32x128 in fp32 needs ~7GB of MPS memory; 40 epochs on the small
# datasets, 15 otherwise; test F1 reported at the best-validation epoch.
# DA operator per dataset follows the repo's run_all_er_magellan.py; --dk general is on.
set -u
cd "$(dirname "$0")/../external/ditto" || exit 1
PY=../../.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
LOGDIR=../../results/ditto
mkdir -p "$LOGDIR"

run() {  # task, da_op, epochs
  local task=$1 op=$2 ep=$3
  local tag=${task//\//_}
  echo "=== $task (da=$op, epochs=$ep) $(date +%T)"
  PYTHONUNBUFFERED=1 nice $PY train_ditto.py \
    --task "$task" --logdir results_ditto/ --finetuning \
    --batch_size 16 --max_len 128 --lr 3e-5 --lm roberta \
    --n_epochs "$ep" --run_id 0 --da "$op" --dk general \
    > "$LOGDIR/$tag.log" 2>&1
  grep -h "^epoch" "$LOGDIR/$tag.log" | tail -1
}

run Structured/Beer            drop_col   40
run Structured/Fodors-Zagats   append_col 40
run Structured/iTunes-Amazon   drop_col   40
run Textual/Abt-Buy            swap       15
run Structured/Walmart-Amazon  drop_col   15
# Amazon-Google and DBLP-ACM are queued last; each takes ~2h on this machine
#run Structured/Amazon-Google   swap       15
#run Structured/DBLP-ACM        swap       15
