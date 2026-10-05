# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
import importlib
import pkgutil

import lynceus.ui.viewers as viewers_package

_registry: dict[str, list[type]] = {}


def register(kind: str, viewer_class: type) -> None:
    """Register a viewer variant for a data kind."""
    _registry.setdefault(kind, []).append(viewer_class)


def viewers_for(kind: str) -> list[type]:
    """Return viewer variants registered for a data kind."""
    return list(_registry.get(kind, []))


def kinds() -> list[str]:
    return list(_registry)


def discover() -> None:
    """Import every viewer module to trigger self-registration.

    Adding a viewer only requires a module in this package that calls
    ``register``; the registry itself does not need to change.
    """
    package_name = viewers_package.__name__
    for module_info in pkgutil.walk_packages(
        viewers_package.__path__, prefix=f"{package_name}."
    ):
        if module_info.name in (
            f"{package_name}.registry",
            f"{package_name}.base",
        ):
            continue
        importlib.import_module(module_info.name)
