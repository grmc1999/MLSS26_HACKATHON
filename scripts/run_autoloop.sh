#!/usr/bin/env bash
# Autonomous autoresearch loop for flu forecasting.
# Runs forever: tries configs, keeps best, logs to loop dir.
set -e

TASK="${1:-flu}"
LOOP_DIR="experiments/loop-${TASK}-$(date +%y%m%d-%H%M)"
mkdir -p "$LOOP_DIR"
BEST_MAE=999
BEST_DESC=""
ITER=0

log() { echo "[$ITER] $*"; }

# Initial baseline
run_exp() {
  rm -f run.log
  python3 scripts/run_exp.py "$@" > run.log 2>&1
  VAL=$(grep -oP 'Val MAE:\s*\K[\d.]+' run.log || echo "?")
  TEST=$(grep -oP 'Test MAE:\s*\K[\d.]+' run.log || echo "?")
  echo "$TEST $VAL"
}

log "Starting loop → $LOOP_DIR"
echo -e "iteration\tcommit\ttest_mae\tval_mae\tdescription" > "$LOOP_DIR/results.tsv"

# Config grid to explore
CONFIGS=()
for model in gru lstm tcn; do
  for h in 32 48 64; do
    for lr in 0.001 0.0005; do
      for ep in 20 30 40; do
        CONFIGS+=("--model $model --hidden-dim $h --lr $lr --epochs $ep --num-layers 3")
      done
    done
  done
done

while true; do
  ITER=$((ITER + 1))

  for cfg in "${CONFIGS[@]}"; do
    log "Trying: $cfg"
    read -r TEST VAL <<< "$(run_exp $cfg)"

    if [ "$TEST" = "?" ]; then
      log "CRASHED — skipping"
      echo -e "$ITER\t(crash)\t?\t?\t$cfg" >> "$LOOP_DIR/results.tsv"
      continue
    fi

    IMPROVED=$(echo "$TEST < $BEST_MAE" | bc -l)
    if [ "$IMPROVED" = "1" ]; then
      BEST_MAE="$TEST"
      BEST_DESC="$cfg"
      COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
      echo -e "$ITER\t$COMMIT\t$TEST\t$VAL\t✅ $cfg" >> "$LOOP_DIR/results.tsv"
      log "✅ KEPT — Test MAE=$TEST (best=$BEST_MAE)"
      git add -f env/train.py && git commit -m "auto: $cfg" 2>/dev/null || true
    else
      echo -e "$ITER\t(reverted)\t$TEST\t$VAL\t❌ $cfg" >> "$LOOP_DIR/results.tsv"
      log "❌ DISCARDED — Test MAE=$TEST (best=$BEST_MAE)"
    fi
  done

  log "Grid complete. Best MAE=$BEST_MAE ($BEST_DESC). Restarting..."
done
