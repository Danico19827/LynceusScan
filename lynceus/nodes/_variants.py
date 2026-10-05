# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Strategy nodes (family + variants) — domain contract, no Qt.

A *strategy node* is a base node whose per-instance contract (INPUTS/OUTPUTS
and PROCESSING_SPECS) depends on a strategy selected in its config
(``STRATEGY_FIELD``, default ``"strategy"``). Choosing a strategy activates a
*variant module*: a plain `.py` that declares ``VARIANT_OF``, ``VARIANT_KEY``,
``VARIANT_LABEL`` and its own INPUTS/OUTPUTS/PROCESSING_SPECS. Variants are
discovered by the registry via AST (never executed at discovery) and imported
lazily when a run or the UI resolves them.

The engine and the UI agree on this resolution contract:

* ``current_variant_key(node_id, config)`` -> the active variant key or None.
* ``effective_ports(node_id, config)``      -> ``(inputs, outputs)``.
* ``effective_specs(node_id, config)``      -> base specs merged with the
  variant's (raw PROCESSING_SPECS, task *names* unresolved).

A strategy node with an empty strategy exposes no ports and no specs: it is
inert until the user picks a variant (the canvas blocks runs in that state).
Non-variant modules are unaffected: ``current_variant_key`` returns None and
everything resolves to the static module contract, so the change is invisible
to existing nodes.
"""

from __future__ import annotations

STRATEGY_FIELD = "strategy"


def current_variant_key(node_id: str, config: dict | None) -> str | None:
    """Return the active variant key for an instance, or None (no strategy)."""
    cfg = config if isinstance(config, dict) else {}
    key = cfg.get(STRATEGY_FIELD)
    if not isinstance(key, str) or not key.strip():
        return None
    return key.strip()


def strategy_options(node_id: str) -> list[tuple[str, str]]:
    """[(key, label)] for a strategy family; [] if the node has no variants."""
    from lynceus.plugins.locale import t
    from lynceus.plugins.registry import manager

    return [
        (item["key"], t(item["label"]))
        for item in manager.list_variants(node_id)
    ]


def variant_meta(node_id: str, key: str) -> dict:
    """AST metadata of a variant (file filters, target row, ...) or {}."""
    from lynceus.plugins.registry import manager

    return manager.variant_meta(node_id, key) or {}


def variant_module(node_id: str, key: str):
    """Import (and cache) the variant module, or None if it cannot be loaded."""
    from lynceus.plugins.registry import manager

    if not key:
        return None
    try:
        return manager.import_variant(node_id, key)
    except Exception:
        return None


def effective_ports(node_id: str, config: dict | None) -> tuple[tuple, tuple]:
    """Active-variant ``(inputs, outputs)``; empty when no strategy is set."""
    key = current_variant_key(node_id, config)
    module = variant_module(node_id, key) if key else None
    if module is None:
        return (), ()
    return (
        tuple(getattr(module, "INPUTS", ())),
        tuple(getattr(module, "OUTPUTS", ())),
    )


def _static_ports(module) -> tuple[tuple, tuple]:
    return (
        tuple(getattr(module, "INPUTS", ())),
        tuple(getattr(module, "OUTPUTS", ())),
    )


def resolved_ports(node_id: str, config: dict | None) -> tuple[tuple, tuple]:
    """Per-instance ``(inputs, outputs)`` with static fallback.

    Variant nodes resolve to the selected variant's ports; a node without a
    strategy (or non-variant modules) falls back to the module contract.
    """
    key = current_variant_key(node_id, config)
    module = variant_module(node_id, key) if key else None
    if module is not None:
        return _static_ports(module)
    base = _import_base(node_id)
    return _static_ports(base) if base is not None else ((), ())


def _import_base(node_id: str):
    """Import a node module by its id (builtin path or registry import)."""
    try:
        return __import__(node_id, fromlist=["PROCESSING_SPECS"])
    except Exception:
        try:
            from lynceus.plugins.registry import manager

            return manager.import_node(node_id)
        except Exception:
            return None


def effective_specs(node_id: str, config: dict | None) -> dict:
    """Per-instance raw PROCESSING_SPECS (base merged with the variant's).

    Variant keys win over the base so a strategy module fully owns its
    capabilities (barrier_task, session_file, output_globs, qml, ...).
    """
    base_module = _import_base(node_id)
    merged = (
        dict(getattr(base_module, "PROCESSING_SPECS", {}))
        if base_module is not None
        else {}
    )
    key = current_variant_key(node_id, config)
    module = variant_module(node_id, key) if key else None
    if module is None:
        return merged
    merged.update(dict(getattr(module, "PROCESSING_SPECS", {})))
    return merged


__all__ = [
    "STRATEGY_FIELD",
    "current_variant_key",
    "strategy_options",
    "variant_meta",
    "variant_module",
    "effective_ports",
    "resolved_ports",
    "effective_specs",
]