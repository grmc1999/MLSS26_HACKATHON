# autoresearch — Flu Forecasting

This is an experiment to have the LLM do its own research on cross-country ILI forecasting using diffusion models.

## Experiment Protocol

Each experiment runs on a single GPU. The training script runs for a fixed budget.

**What you CAN do:**
- Modify `env/train.py` — model architecture, optimizer, hyperparameters, loss function, etc.

**What you CANNOT do:**
- Modify `env/eval.py` or `env/data.py`.
- Install new packages.

## The experiment loop

1. Modify `env/train.py` with an experimental idea.
2. `git commit`
3. Run: `python scripts/run_flu_pipeline.py --pretrain-epochs 30 --finetune-epochs 10 > run.log 2>&1`
4. Extract: `grep "Test MAE" run.log | awk '{print $NF}'`
5. Record in `results.tsv`
6. If Test MAE improved, keep the commit. Otherwise, revert.
7. Loop until manually stopped.
