---
name: autoresearch
description: "Autonomous iteration loop for flu forecasting: modify, verify, keep/discard"
version: 2.2.0
---

# Autoresearch — Flu Forecasting

- Modify only: `env/train.py`
- Run: `python scripts/run_flu_pipeline.py --pretrain-epochs 30 --finetune-epochs 10 > run.log 2>&1`
- Metric: `grep "Test MAE" run.log | awk '{print $NF}'`
- Log to: `experiments/loop-flu-{YYMMDD}-{HHMM}/results.tsv`
