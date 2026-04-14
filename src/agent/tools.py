"""Tool specification and registry.

A ``ToolSpec`` bundles a JSON schema (for the chat template) with a
Python callable (for actual execution). The Agent holds a
``ToolRegistry`` that maps tool names to specs and dispatches tool
calls from the model's parsed output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ToolFunction = Callable[[dict[str, Any]], dict[str, Any]]
ToolSchemaFormatter = Callable[[str, str, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """A tool exposed to the agent.

    Attributes:
        name: Tool name as it appears in the chat template.
        description: Human-readable description for the model.
        parameters: JSON schema describing the tool's arguments.
        fn: Python callable that implements the tool. Takes the
            argument dict and returns a JSON-serialisable result dict.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFunction

    def render_schema(self, formatter: ToolSchemaFormatter) -> dict[str, Any]:
        """Render this tool's schema using a profile-provided formatter.

        Each model family expects a different shape in its ``<tools>``
        block, so the formatter comes from the active ``ModelProfile``
        rather than being hardcoded here.

        Args:
            formatter: A callable taking ``(name, description, parameters)``
                and returning the family-specific dict.

        Returns:
            The family-specific tool schema dict.
        """
        return formatter(self.name, self.description, self.parameters)


@dataclass
class ToolRegistry:
    """A name → ``ToolSpec`` registry with a dispatch helper.

    Attributes:
        specs: Mapping from tool name to ``ToolSpec``.
    """

    specs: dict[str, ToolSpec] = field(default_factory=dict)

    @classmethod
    def from_list(cls, tools: list[ToolSpec]) -> "ToolRegistry":
        """Build a registry from a flat list of specs.

        Args:
            tools: Tool specs to register.

        Returns:
            A populated ``ToolRegistry``.

        Raises:
            ValueError: If two tools share the same name.
        """
        seen: dict[str, ToolSpec] = {}
        for spec in tools:
            if spec.name in seen:
                raise ValueError(f"Duplicate tool name: {spec.name!r}")
            seen[spec.name] = spec
        return cls(specs=seen)

    def schemas(self, formatter: ToolSchemaFormatter) -> list[dict[str, Any]]:
        """Return the list of schemas for all registered tools.

        Args:
            formatter: Family-specific schema formatter from the active
                ``ModelProfile``.

        Returns:
            A list of dicts, one per registered tool, shaped for the
            family's chat template.
        """
        return [spec.render_schema(formatter) for spec in self.specs.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name.

        Args:
            name: Tool name from the model's parsed output.
            arguments: Argument dict from the model's parsed output.

        Returns:
            The tool's result dict.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self.specs:
            raise KeyError(f"Unknown tool: {name!r}")
        return self.specs[name].fn(arguments)
