# autoresearch — Dynamical Systems Forecasting (Flu / RESPNET / ILINET)

This is an experiment to have the LLM do its own research on forecasting flu and respiratory illness dynamics from RESPNET and ILINET surveillance data.

## Task Setup

- **Input**: 5 past epiweeks of RESPNET/ILINET observations (weekly %ILI rates)
- **Output**: 10 future epiweeks forecast (multi-step ahead)
- **Data**: CDC ILINET surveillance network (HHS regions or national)
- **Metric**: MAE (mean absolute error) on the 10-step-ahead forecast; lower is better
- **Baseline**: seasonal naive forecast (repeat last season's values)

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `env/train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `env/data.py` — RESPNET/ILINET data loader.
   - `AGENTS.md` — agent system documentation and model configurations.
   - `configs/agents.yaml` — agent role → model mappings.
4. **Verify data exists**: Check that `env/data/` contains the RESPNET/ILINET data.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (RTX PRO 6000, 98GB VRAM). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, loss function, data pipeline, etc.
- Use any of the specialized agents (cv_expert, dl_expert, satellite_expert, physics_expert) for domain-specific advice.

**What you CANNOT do:**
- Modify `eval.py` or `data.py`. They are read-only. They contain the fixed evaluation and data functions.
- Install new packages or add dependencies. You can only use what's already installed.

**The goal is simple: get the lowest MAE** on the 10-step-ahead forecast. Lower is better. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the loss function. The only constraint is that the code runs without crashing.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful MAE gains, but it should not blow up dramatically. You have 98GB available.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
Test MAE: 0.1234
```

You can extract the key metric from the log file:

```
grep "Test MAE" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	mae	status	description
```

1. git commit hash (short, 7 chars)
2. mae achieved (e.g. 0.1234) — use 999.0 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	mae	status	description
a1b2c3d	0.1234	keep	baseline seasonal naive
b2c3d4e	0.0987	keep	seq2seq LSTM with teacher forcing
c3d4e5f	0.1500	discard	Transformer without positional encoding
d4e5f6g	999.0	crash	OOM with full attention
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jun22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "Test MAE" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If MAE improved (lower), you "advance" the branch, keeping the git commit
9. If MAE is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes total. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Specialized Agent Consultation

When you need domain-specific expertise, consult the specialized agents:

- **CV Expert**: For time series architectures (LSTM, GRU, Transformer, TCN, N-BEATS)
- **DL Expert**: For training loop optimization, loss functions (MAE, RMSE, quantile), optimizer config
- **Satellite Expert**: For RESPNET/ILINET data interpretation, seasonal patterns, external covariates
- **Physics Expert**: For compartmental models (SIR, SEIR), Neural ODEs, physical constraints
- **Continual Learning Expert**: For checkpoint management, anti-forgetting strategies across iterations

## Suggested Experiment Progression

1. **Baseline**: Run the starter code (seasonal naive or linear model) to establish baseline MAE
2. **Seq2Seq LSTM**: Replace with encoder-decoder LSTM with teacher forcing
3. **Loss function**: Try MAE, RMSE, Huber, or quantile loss
4. **Normalization**: Add z-score or min-max normalization per time series
5. **Learning rate schedule**: Add cosine annealing or reduce-on-plateau
6. **Temporal features**: Add Fourier features (sin/cos epiweek encoding), lag features
7. **Architecture**: Try TCN (temporal convolutional network) or Transformer
8. **Multi-horizon strategy**: Compare direct, recursive, and MIMO forecasting
9. **External covariates**: Add temperature, vaccination, mobility as auxiliary inputs
10. **Advanced**: Neural ODE, temporal fusion transformer, deep ensembles
