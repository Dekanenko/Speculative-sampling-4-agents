"""HotPotQA task family.

Wikipedia-backed multi-hop QA tasks. Exports the
:class:`HotpotqaFamily` for registration, plus helpers for the
offline prepare step.
"""

from .env import WikipediaCache
from .evaluator import exact_match, extract_predicted_answer, token_f1
from .loader import BENCHMARKS_ROOT, HotpotqaFamily, SPLIT_NAME, cache_path_for, split_dir
from .tools import RateLimiter, build_wikipedia_tools

__all__ = [
    "BENCHMARKS_ROOT",
    "HotpotqaFamily",
    "RateLimiter",
    "SPLIT_NAME",
    "WikipediaCache",
    "build_wikipedia_tools",
    "cache_path_for",
    "exact_match",
    "extract_predicted_answer",
    "split_dir",
    "token_f1",
]
