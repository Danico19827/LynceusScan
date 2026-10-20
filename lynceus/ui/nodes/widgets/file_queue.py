# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""File-queue behavior shared by every file-backed node item (Qt side).

The queue itself (normalize/add/remove/move/partition/summarize) lives in
the Qt-free base ``lynceus.nodes._file_queue``; this mixin binds it to a
painted item. One source of truth: ``_file_queue`` holds ALL files (empty
= none, one = single file, several = queue) and the visible ``_file_path``
/``_file_name``/``_file_missing`` always mirror its head. Persistence
shape differs per family (flat ``config`` for Load, ``config.pipeline``
for product inputs), so each item implements ``_persist_queue`` /
``_restore_queue``; everything else is shared.

``supports_file_queue`` gates the "Manage queue..." context-menu entry:
only the three file-picking items opt in (strategy subclasses without a
Browse button explicitly opt out).
"""

import os

from lynceus.nodes import _file_queue as fq


class FileQueueMixin:
    """Queue state + multi-file Browse for a file-picking node item."""

    supports_file_queue = True

    def _queue_init(self) -> None:
        self._file_queue: list[str] = []

    # ---------- state ----------

    def file_queue(self) -> list[str]:
        """Queued files in order (empty = single-file mode)."""
        return list(getattr(self, "_file_queue", []))

    def set_file_queue(self, paths) -> bool:
        """Replace the queue; the head becomes the effective file."""
        queue = fq.set_queue(paths)
        changed = queue != self.file_queue()
        self._file_queue = queue
        self._sync_queue_state()
        if changed:
            self.file_changed.emit()
            self.update()
        return changed

    def _sync_queue_state(self) -> None:
        """Mirror the queue head into the visible file attributes."""
        queue = self.file_queue()
        path = queue[0] if queue else None
        self._file_path = path
        self._file_name = os.path.basename(path) if path else None
        self._file_missing = bool(path) and not os.path.isfile(path)
        self._persist_queue(queue)

    def _persist_queue(self, queue: list[str]) -> None:
        """Store the queue in this family's config shape (per item)."""
        raise NotImplementedError

    def _restore_queue(self, files, fallback: str | None) -> None:
        """Rebuild state from persisted config (no signals on restore)."""
        from lynceus.nodes._file_queue import normalize_queue

        queue = normalize_queue(files) if isinstance(files, list) else []
        if not queue and isinstance(fallback, str) and fallback:
            queue = [fallback]
        self._file_queue = queue
        path = queue[0] if queue else None
        self._file_path = path
        self._file_name = os.path.basename(path) if path else None
        self._file_missing = bool(path) and not os.path.isfile(path)

    # ---------- browsing ----------

    def _browse_files(self):
        """Multi-file picker; override per item for dialog filters."""
        raise NotImplementedError

    def _browse(self) -> None:
        paths = self._browse_files()
        if paths:
            self.set_file_queue(paths)

    # ---------- labels ----------

    def _queue_label_text(self, no_file_text: str) -> tuple[str, bool]:
        """(label text, missing) honoring single vs. queued display."""
        from lynceus.plugins.locale import t

        queue = self.file_queue()
        if len(queue) > 1:
            return fq.queue_count_text(len(queue)), False
        if self._file_name:
            if self._file_missing:
                return (
                    f"\u26a0 {self._file_name} {t('(file not found)')}",
                    True,
                )
            return f"\u2713 {self._file_name}", False
        return no_file_text, False
