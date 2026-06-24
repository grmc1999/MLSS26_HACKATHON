I have `autoresearch_pipeline.md`. Do **not** edit `md` yet. I only want you to create the reusable code scripts that make the pipeline more reliable across many iterations.

The problem: the current pipeline is long, and by iteration 3 or 4 the LLM sometimes forgets to run RAG, create the required JSON log, or fill all fields. I want scripts that enforce those steps deterministically.

See `scripts/README_scripts.md` for usage.