I have `autoresearch_pipeline.md`. Do **not** edit `md` yet. I only want you to create the reusable code scripts that make the pipeline more reliable across many iterations.

The problem: the current pipeline is long, and by iteration 3 or 4 the LLM sometimes forgets to run RAG, create the required JSON log, or fill all fields. I want scripts that enforce those steps deterministically.

Please create a `scripts/` directory with these files:

1. `run_rag_search.py`
2. `make_iter_log.py`
3. `append_results_tsv.py`
4. `code_jury.py`
5. `pipeline_utils.py`

Do not install new packages unless absolutely necessary. Prefer Python standard library. Assume the repo already contains the RAG functions mentioned in the markdown:

* For medmnist: `search_medical_literature(query, k=5, task="medmnist")`
* For flu: `search_flu_context_rag(query, k=5)`

If those functions are not directly importable, make the import path configurable with CLI flags such as `--rag-module` and implement clear fallback error messages.

## Desired script behavior

### 1. `run_rag_search.py`

Purpose: run mandatory literature/RAG search every iteration and save evidence to JSON.

CLI example:

```bash
python scripts/run_rag_search.py \
  --task flu \
  --iteration 4 \
  --query "cross-country ILI forecasting domain adaptation dropout regularization" \
  --k 5 \
  --out experiments/loop-flu-260623-2130/iterations/iter-4-rag.json
```

Requirements:

* Accept `--task` with choices: `medmnist`, `flu`.
* Accept `--iteration`, `--query`, `--k`, `--out`.
* Optional: `--rag-module`.
* For `medmnist`, call `search_medical_literature(query, k=k, task="medmnist")`.
* For `flu`, call `search_flu_context_rag(query, k=k)`.
* Save a JSON object with this shape:

```json
{
  "iteration": 4,
  "task": "flu",
  "query": "...",
  "k": 5,
  "timestamp": "ISO-8601 UTC timestamp",
  "results": [
    {
      "title": "...",
      "source": "...",
      "score": 0.0,
      "snippet": "...",
      "metadata": {}
    }
  ]
}
```

* Normalize whatever the RAG function returns into this schema as best as possible.
* Never silently succeed with empty results. If results are empty, write the file but mark `"warning": "rag returned zero results"` and exit with a nonzero code unless `--allow-empty` is passed.
* Create parent directories automatically.

### 2. `make_iter_log.py`

Purpose: generate the mandatory per-iteration JSON log and validate that all required fields are filled.

CLI example:

```bash
python scripts/make_iter_log.py \
  --task flu \
  --iteration 4 \
  --status keep \
  --metric-name "Test MAE" \
  --metric-direction lower_is_better \
  --metric-value 0.5823 \
  --baseline 0.5897 \
  --change-type regularization \
  --change-file env/train.py \
  --diff-summary "Added dropout 0.2 between GRU encoder layers" \
  --hypothesis "Dropout will reduce overfitting to US-specific patterns and improve cross-country generalization." \
  --mechanism "nn.Dropout(0.2) is applied to encoder hidden states before decoding, forcing redundant representations." \
  --expected-delta "-0.01 to -0.03 Test MAE" \
  --risk "low — dropout only active during training" \
  --baseline-compared-to "env.baseline/" \
  --elapsed-s 42.3 \
  --memory-gb 1.2 \
  --params 298881 \
  --commit-hash abc1234 \
  --commit-message "pipeline: time_series_expert — Add dropout 0.2 to GRUSeq2Seq" \
  --out experiments/loop-flu-260623-2130/iterations/iter-4.json
```

Requirements:

* Output exactly the JSON schema shown in the pipeline markdown.
* Required sections:

  * `iteration`
  * `task`
  * `commit`
  * `timestamp`
  * `status`
  * `metric`
  * `change`
  * `jury_reasoning`
  * `resources`
  * `git`
* Refuse empty strings, `"N/A"`, `"unknown"`, or nulls in required fields.
* Validate status is one of: `keep`, `discard`, `crash`.
* Validate direction is one of: `higher_is_better`, `lower_is_better`.
* Create parent directories automatically.
* Pretty-print JSON with indentation.
* Exit nonzero with a clear error if validation fails.

### 3. `append_results_tsv.py`

Purpose: append the human-readable result row to `results.tsv`.

CLI example:

```bash
python scripts/append_results_tsv.py \
  --task flu \
  --results experiments/loop-flu-260623-2130/results.tsv \
  --iteration 4 \
  --commit abc1234 \
  --test-mae 0.5823 \
  --val-mae 0.6011 \
  --params 298881 \
  --memory-gb 1.2 \
  --status keep \
  --description "Added dropout 0.2 between GRU encoder layers"
```

Requirements:

* If the TSV does not exist, create it and write the correct header.
* For `flu`, use columns:
  `iteration, commit, test_mae, val_mae, params, memory_gb, status, description`
* For `medmnist`, use columns:
  `iteration, commit, test_acc, ood_f1, val_acc, test_acc_id, memory_gb, status, description`
* Accept missing optional metric columns but write blank fields rather than crashing.
* Do not duplicate a row if the same iteration already exists unless `--overwrite` is passed.
* Use tab separation, not commas.

### 4. `code_jury.py`

Purpose: run mandatory code validation before commit.

CLI example:

```bash
python scripts/code_jury.py \
  --task flu \
  --env-dir env \
  --train-py env/train.py \
  --input-shape "(4, 5, 1)" \
  --expected-output-shape "(4, 10)" \
  --out experiments/loop-flu-260623-2130/iterations/iter-4-jury.json
```

For medmnist:

```bash
python scripts/code_jury.py \
  --task medmnist \
  --env-dir MLAgentBench/benchmarks/medmnist/env \
  --train-py MLAgentBench/benchmarks/medmnist/env/train.py \
  --input-shape "(4, 1, 28, 28)" \
  --expected-output-shape "(4, 3)" \
  --out experiments/loop-medmnist-260623-2130/iterations/iter-4-jury.json
```

Requirements:

* Run syntax check using `py_compile`.
* Try to import `train.py`.
* Try to discover and instantiate the model. Because model names may vary, implement multiple strategies:

  1. If `--model-factory` is passed, call that function from the imported module.
  2. Else try common names: `get_model`, `build_model`, `create_model`, `Model`, `Net`, `GRUSeq2Seq`, `CNN`, `Classifier`.
  3. If no model can be found, fail with a clear message explaining how to pass `--model-factory`.
* Create random input tensor using PyTorch.
* Run forward pass.
* Verify output shape when `--expected-output-shape` is provided.
* Compute a simple loss:

  * For classification / medmnist: use `torch.nn.CrossEntropyLoss` with random class labels.
  * For regression / flu: use `torch.nn.MSELoss` with random target matching output shape.
* Verify finite loss.
* Run backward pass.
* Verify at least one trainable parameter has a non-null finite gradient.
* Save a JSON report:

```json
{
  "task": "flu",
  "timestamp": "...",
  "status": "pass",
  "checks": {
    "syntax": {"status": "pass"},
    "import": {"status": "pass"},
    "instantiate": {"status": "pass", "model_class": "..."},
    "forward": {"status": "pass", "output_shape": [4, 10]},
    "loss": {"status": "pass", "value": 0.123},
    "backward": {"status": "pass", "params_with_grad": 42}
  },
  "errors": []
}
```

* If any check fails, write the JSON report with `"status": "fail"` and exit nonzero.
* Create parent directories automatically.
* Do not modify any training files.

### 5. `pipeline_utils.py`

Purpose: shared helpers.

Include:

* `utc_now_iso()`
* `ensure_parent_dir(path)`
* `write_json(path, obj)`
* `read_json(path)`
* `reject_empty_required(value, field_name)`
* `run_command(cmd, cwd=None, timeout=None)`
* `parse_shape("(4, 5, 1)") -> tuple[int, ...]`
* `safe_float(value)`
* `safe_int(value)`
* `get_git_commit_hash()` if possible
* `get_git_status_dirty()` if possible

## Quality requirements

* Use `argparse`.
* Use type hints where reasonable.
* Include clear error messages.
* Scripts should be executable as normal Python files.
* Avoid fragile hardcoded absolute paths.
* Include docstrings at the top of each script explaining purpose and example usage.
* Add `if __name__ == "__main__": main()`.
* Use only standard library except `code_jury.py`, which may require PyTorch because the training code already uses PyTorch.
* Include a small `README_scripts.md` explaining how the scripts fit into the pipeline.

## Deliverables

Please output:

1. The complete code for each script.
2. The proposed file tree.
3. Example commands for flu and medmnist.
4. Notes on where I may need to customize import paths for the RAG functions or model factory.

Important: do not rewrite the whole Skill yet. Focus only on making these scripts.
