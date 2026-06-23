"""LLM interface — all calls now handled by opencode.

Previously routed through OpenRouter. LLM capabilities are provided by the
opencode CLI environment this project runs under. See the experiment pipeline
in env/train.py for non-LLM experiment code.

This file is kept as a stub so imports don't break, but all LLM-dependent
agent/inference features have been removed.
"""
from .schema import LLMError


def complete_text(prompt, log_file, model, **kwargs):
    raise LLMError("LLM calls are handled by opencode, not direct API calls")


def complete_text_fast(prompt, **kwargs):
    raise LLMError("LLM calls are handled by opencode, not direct API calls")
