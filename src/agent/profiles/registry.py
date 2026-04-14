"""Profile registry: name → factory function mapping.

Profiles are stored as *factory functions*, not as resolved
``ModelProfile`` instances, because delimiter token IDs depend on the
tokenizer and must be resolved at load time.
"""

from __future__ import annotations

from typing import Any, Callable

from . import llama3, qwen3, qwen25
from .base import ModelProfile


ProfileFactory = Callable[[Any], ModelProfile]


_REGISTRY: dict[str, ProfileFactory] = {
    "qwen2.5": qwen25.build,
    "qwen3": qwen3.build,
    "llama3": llama3.build,
}


def register_profile(name: str, factory: ProfileFactory) -> None:
    """Register a new profile factory under ``name``.

    Args:
        name: Short family name (e.g., ``"mistral"``).
        factory: A callable taking a tokenizer and returning a ``ModelProfile``.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Profile {name!r} is already registered")
    _REGISTRY[name] = factory


def build_profile(name: str, tokenizer: Any) -> ModelProfile:
    """Resolve a profile by name against a concrete tokenizer.

    Args:
        name: Registered profile name.
        tokenizer: Tokenizer whose vocabulary will be used to resolve
            delimiter token IDs.

    Returns:
        A fully-resolved ``ModelProfile``.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown profile {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](tokenizer)


def list_profiles() -> list[str]:
    """Return the sorted list of registered profile names."""
    return sorted(_REGISTRY)
