"""Task family registry.

Each family bundles a ``load_tasks`` function, per-task ``build_env``
and ``build_tools`` factories, and an ``evaluate`` method into a
single object. The runner uses :func:`get_family` to look one up by
name and drives the agent loop through its interface.
"""

from __future__ import annotations

from typing import Callable

from .base import EvaluationResult, Env, TaskFamily
from .mocks.family import MocksFamily


_REGISTRY: dict[str, TaskFamily] = {}


def register_family(family: TaskFamily) -> None:
    """Register a task family under its ``name``.

    Args:
        family: Family instance to register.

    Raises:
        ValueError: If a family with the same name is already registered.
    """
    if family.name in _REGISTRY:
        raise ValueError(f"Family {family.name!r} is already registered")
    _REGISTRY[family.name] = family


def get_family(name: str) -> TaskFamily:
    """Look up a registered family by name.

    Args:
        name: Family name (e.g. ``"mocks"``, ``"hotpotqa"``).

    Returns:
        The registered ``TaskFamily`` instance.

    Raises:
        KeyError: If no family with that name is registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown family {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_families() -> list[str]:
    """Return the sorted list of registered family names."""
    return sorted(_REGISTRY)


# Built-in families.
register_family(MocksFamily())


__all__ = [
    "EvaluationResult",
    "Env",
    "TaskFamily",
    "get_family",
    "list_families",
    "register_family",
]
