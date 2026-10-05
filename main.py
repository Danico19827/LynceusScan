# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
import multiprocessing
import sys
import time as _time

from PySide6.QtWidgets import QApplication, QDialog, QSplashScreen

from lynceus import project as project_io
from lynceus.plugins import terms
from lynceus.plugins.registry import manager
from lynceus.plugins.store import ConsentStore
from lynceus.ui.branding import (
    COPYRIGHT_LINE,
    set_splash_progress,
    splash_base_pixmap,
)
from lynceus.ui.main_window import MainWindow


def _setup_theme() -> None:
    """Active theme + font from settings (core dark, system font)."""
    from lynceus.ui.fonts import apply_font
    from lynceus.ui.theming import apply_theme

    apply_theme()
    apply_font()


def _require_terms(store: ConsentStore) -> None:
    """First-launch gate: exit unless the user agrees to the Terms & Conditions.

    Runs before the main window is created, so declining never shows the
    application. Applies to every installation (including development).
    """
    from lynceus.plugins.locale import locale_manager
    from lynceus.ui.terms_dialog import TermsDialog

    blob = terms.terms_blob()
    if store.terms_accepted(terms.TERMS_VERSION, blob):
        return
    accepted = (
        TermsDialog(
            terms.TERMS_VERSION, terms.terms_text(locale_manager.language)
        ).exec()
        == QDialog.DialogCode.Accepted
    )
    if accepted:
        store.terms_accept(terms.TERMS_VERSION, blob)
    else:
        sys.exit(0)


def _setup_language() -> None:
    """Active language from settings (English by default)."""
    from PySide6.QtCore import QSettings

    from lynceus.plugins.locale import locale_manager
    from lynceus.ui.main_window import SETTINGS_APP, SETTINGS_ORG

    saved = QSettings(SETTINGS_ORG, SETTINGS_APP).value("language", "en")
    locale_manager.set_language(saved or "en")


def main():
    # Frozen first runs: seed bundled packs (themes) into the user
    # extensions dir BEFORE anything can trigger extension discovery
    # (the theme setup below already discovers). Otherwise the first
    # run caches an empty registry and the packs never enable.
    # No-op in source checkouts. No Qt needed.
    from lynceus.plugins.importer import seed_bundled_extensions

    seed_bundled_extensions()

    app = QApplication(sys.argv)
    # Window/taskbar icon is themed inside _setup_theme (apply_theme);
    # all windows inherit the app icon.
    _setup_theme()

    splash = QSplashScreen(splash_base_pixmap())
    splash.show()
    app.processEvents()
    splash_started = _time.monotonic()

    store = ConsentStore()
    manager.disabled_provider = lambda: store.disabled_ids()

    _setup_language()
    from lynceus.plugins.locale import t

    base = splash.pixmap()
    set_splash_progress(splash, base, 0.2, t("Loading interface\u2026"))
    _require_terms(store)

    set_splash_progress(splash, base, 0.55, t("Discovering nodes\u2026"))
    window = MainWindow()
    set_splash_progress(splash, base, 0.85, t("Preparing workspace\u2026"))
    _wait_splash_floor(app, splash_started)
    set_splash_progress(splash, base, 1.0, COPYRIGHT_LINE)
    # The launcher is modal and would sit on top of the splash: close the
    # splash first so it is seen on its own, then offer the launcher.
    splash.close()
    if "--smoke-test" in sys.argv:
        # Frozen-build probe: blank canvas, no modal launcher, quit after
        # the window paints. The runner pre-seeds T&C consent in a temp
        # %APPDATA% so the terms gate passes unattended. Reports the
        # actually-enabled packs so frozen availability (not just files)
        # is verified: discovery order bugs show up here.
        import json

        from PySide6.QtCore import QTimer

        from lynceus.plugins.locale import locale_manager
        from lynceus.plugins.store import data_root
        from lynceus.plugins.theme import theme_manager

        report = {
            "themes": theme_manager.available(),
            "languages": locale_manager.available(),
            "extensions_dir": str(manager.extensions_dir),
        }
        (data_root() / "smoke_result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        window.show()
        window.maximize_window()
        QTimer.singleShot(4000, app.quit)
        code = app.exec()
        print("SMOKE-OK")
        sys.exit(code)
    _open_startup_selection(window)

    window.show()
    window.maximize_window()

    sys.exit(app.exec())


def _wait_splash_floor(app, started: float) -> None:
    """Hold the splash for at least SPLASH_MIN_SECONDS (event loop pumped).

    Runs BEFORE the startup launcher: that dialog is modal and would cover
    the splash, so waiting after it means the floor elapses unseen behind it.
    """
    from lynceus.ui.branding import SPLASH_MIN_SECONDS

    while True:
        remaining = SPLASH_MIN_SECONDS - (_time.monotonic() - started)
        if remaining <= 0:
            break
        _time.sleep(min(0.05, remaining))
        app.processEvents()


def _cli_project_path() -> str | None:
    """An optional .lynx path passed on the command line."""
    for arg in sys.argv[1:]:
        if arg.lower().endswith(project_io.PROJECT_SUFFIX):
            from pathlib import Path

            if Path(arg).is_file():
                return arg
    return None


def _open_startup_selection(window: MainWindow) -> None:
    """Launcher before the main window (blank/open/recent/template)."""
    from PySide6.QtCore import QSettings

    from lynceus import templates as templates_io
    from lynceus.ui.settings_keys import SETTINGS_APP, SETTINGS_ORG, SHOW_START_KEY
    from lynceus.ui.start_dialog import StartDialog

    cli_path = _cli_project_path()
    if cli_path is not None:
        window._open_path(cli_path)
        return

    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    if not settings.value(SHOW_START_KEY, True, type=bool):
        return  # blank canvas, current startup behavior

    dialog = StartDialog(
        window._load_recent(),
        templates_io.list_templates(),
        show_on_startup=True,
    )
    if dialog.exec() == QDialog.DialogCode.Accepted:
        kind, path, from_template = dialog.action
        if kind == "open" and path:
            window._open_path(path, from_template=from_template)
    # Rejected/closed without choosing: blank canvas.
    settings.setValue(SHOW_START_KEY, dialog.show_on_startup)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
