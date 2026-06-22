# autoresearch — Chest X-ray OOD Detection

This is an experiment to have the LLM do its own research on out-of-distribution (OOD) detection for chest X-ray classification using MedMNIST.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The task is small. Read these files for full context:
   - `MLAgentBench/benchmarks/medmnist/env/data_description.txt` — dataset description.
   - `MLAgentBench/benchmarks/medmnist/env/train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `MLAgentBench/benchmarks/medmnist/env/evaluation_details.txt` — evaluation details.
   - `AGENTS.md` — agent system documentation and model configurations.
   - `configs/agents.yaml` — agent role → model mappings.
4. **Verify data exists**: Check that `MLAgentBench/benchmarks/medmnist/env/train/` contains the training data. If not, tell the human to run the MedMNIST download.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (RTX PRO 6000, 98GB VRAM). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python scripts/run_medmnist.py`.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, loss function, data augmentation, calibration method, OOD detection strategy, etc.
- Use any of the specialized agents (cv_expert, dl_expert, medical_expert, robustness_expert) for domain-specific advice.

**What you CANNOT do:**
- Modify `eval.py` or `encode.py`. They are read-only. They contain the fixed evaluation and encoding functions.
- Install new packages or add dependencies. You can only use what's already installed.
- Modify the evaluation harness. The accuracy and OOD F1 functions are the ground truth metrics.

**The goal is simple: get the highest Test Accuracy and OOD F1.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the loss function, add temperature scaling, Mahalanobis scoring, ensembles. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful metric gains, but it should not blow up dramatically. You have 98GB available.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 accuracy improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
Test Accuracy: 0.2200
OOD F1: 0.1500
```

You can extract the key metrics from the log file:

```
grep "Test Accuracy" run.log
grep "OOD F1" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	test_acc	ood_f1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. test_acc achieved (e.g. 0.2200) — use 0.0000 for crashes
3. ood_f1 achieved (e.g. 0.1500) — use 0.0000 for crashes
4. peak memory in GB, round to .1f (e.g. 1.2)
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	test_acc	ood_f1	memory_gb	status	description
a1b2c3d	0.2200	0.1500	1.2	keep	baseline single conv layer
b2c3d4e	0.4500	0.3200	2.5	keep	upgrade to 4-layer CNN with dropout
c3d4e5f	0.2100	0.1400	1.2	discard	switch to Focal loss only
d4e5f6g	0.0000	0.0000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jun22` or `autoresearch/jun22-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python scripts/run_medmnist.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "Test Accuracy" run.log` and `grep "OOD F1" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If either Test Accuracy or OOD F1 improved, you "advance" the branch, keeping the git commit
9. If both metrics are equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Specialized Agent Consultation

When you need domain-specific expertise, consult the specialized agents:

- **CV Expert**: For architecture design (CNN, ResNet, Vision Transformer), data augmentation, preprocessing for 28×28 images
- **DL Expert**: For training loop optimization, loss functions (CE, Focal), optimizer config, temperature scaling
- **Medical Expert**: For chest X-ray pathology knowledge, MedMNIST dataset understanding
- **Robustness Expert**: For OOD detection methods (Mahalanobis, energy score, ODIN, ensemble uncertainty), confidence calibration
- **Continual Learning Expert**: For checkpoint management, anti-forgetting strategies across iterations

## Suggested Experiment Progression

1. **Baseline**: Run the starter code (single conv layer) to establish baseline Test Accuracy and OOD F1
2. **Deep CNN**: Replace with a proper CNN (3-4 conv layers with batch norm, dropout, max pooling)
3. **Training optimization**: Tune learning rate, batch size, weight decay, add learning rate scheduling
4. **Data augmentation**: Add rotations, flips, random affine transforms
5. **Confidence calibration**: Add temperature scaling to the output logits
6. **OOD detection**: Implement Mahalanobis distance-based OOD detection using feature embeddings
7. **Energy-based OOD**: Replace softmax confidence with energy score for OOD detection
8. **Ensemble**: Train a small ensemble (3-5 models) with Monte Carlo dropout or deep ensembles
9. **ODIN**: Implement ODIN with temperature scaling + input preprocessing
10. **Advanced architectures**: Try ResNet-18/34, Vision Transformer (ViT) for 28×28 images
