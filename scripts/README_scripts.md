# Pipeline Scripts

Deterministic helpers for the `autoresearch_pipeline.md` experiment loop.

## File Tree

```
scripts/
├── pipeline_utils.py      # Shared helpers (JSON, git, shape parsing)
├── run_rag_search.py      # Phase 1: RAG literature search → JSON
├── make_iter_log.py       # Phase 8: per-experiment JSON log with validation
├── append_results_tsv.py  # Phase 8: append human-readable row to results.tsv
├── code_jury.py           # Phase 4b: syntax + forward + loss + backward checks
└── README_scripts.md      # This file
```

## How They Fit Into the Pipeline

### Phase 1 — Research
```bash
python scripts/run_rag_search.py \
  --task flu \
  --iteration 4 \
  --query "cross-country ILI forecasting" \
  --k 5 \
  --out experiments/loop-flu-260623-2130/iterations/iter-4-rag.json
```

### Phase 4b — Code Jury
```bash
python scripts/code_jury.py \
  --task flu \
  --env-dir env \
  --train-py env/train.py \
  --input-shape "(4, 5, 1)" \
  --expected-output-shape "(4, 10)" \
  --out experiments/loop-flu-260623-2130/iterations/iter-4-jury.json
```

### Phase 8 — Log
```bash
python scripts/make_iter_log.py \
  --task flu --iteration 4 --status keep \
  --metric-name "Test MAE" --metric-direction lower_is_better \
  --metric-value 0.5823 --baseline 0.5897 \
  --change-type regularization --change-file env/train.py \
  --diff-summary "Added dropout 0.2 between GRU encoder layers" \
  --hypothesis "..." --mechanism "..." --expected-delta "..." \
  --risk "low" --baseline-compared-to "env.baseline/" \
  --elapsed-s 42.3 --memory-gb 1.2 --params 298881 \
  --commit-hash abc1234 \
  --commit-message "pipeline: time_series_expert -- Add dropout" \
  --out experiments/loop-flu-260623-2130/iterations/iter-4.json

python scripts/append_results_tsv.py \
  --task flu \
  --results experiments/loop-flu-260623-2130/results.tsv \
  --iteration 4 --commit abc1234 --test-mae 0.5823 \
  --val-mae 0.6011 --params 298881 --memory-gb 1.2 \
  --status keep --description "Added dropout 0.2 between GRU encoder layers"
```

## Customization Points

### RAG Import Path
The scripts attempt `from MLAgentBench.agents.agent_specialized import ...`.
If your RAG functions live elsewhere, pass `--rag-module my.custom.module`.

### Model Factory
`code_jury.py` tries common model class/function names automatically.
If your model uses an unusual name, pass `--model-factory build_my_model`.
