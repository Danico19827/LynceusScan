# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Application branding: window icon, menu logo and startup splash (Qt).

All brand assets flow through this module so a future theme only touches
``THEME_COLORS`` (and optionally passes its own ``logo_path``): no color
or asset literal lives at the call sites.

Caveat: the Inkscape SVGs in ``assets/`` use millimetre units, which makes
``QIcon(svg).pixmap()`` come back null. Every pixmap here is rendered with
``QSvgRenderer`` at explicit pixel sizes instead of relying on ``QIcon``
auto-loading.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import QSplashScreen
from PySide6.QtSvg import QSvgRenderer

from lynceus import __version__ as APP_VERSION
from lynceus.resources import resource_path
from lynceus.ui.fonts import scaled_point_size

ICON_SQUARE_SVG = str(
    resource_path("assets/Icono Transparente LynceusScan.svg")
)
SPLASH_LOCKUP_PNG = str(resource_path("assets/splash_lockup.png"))

# Theme contract: the main background surfaces of the active theme.
# Today this mirrors the core dark QSS (deep window background); a future
# theme overrides these values and the splash/chrome follow automatically.
THEME_COLORS = {
    "splash_bg": "#0b0f1a",  # QSS deep background (window, canvas)
    "surface": "#1c2333",  # QSS surfaces (panels, menus, titlebars)
    "border": "#2f3a4f",  # QSS borders/selection
    "text": "#e6e9f0",  # QSS primary text
    "muted": "#8a93a6",  # QSS secondary text
    "accent": "#7d9fd4",  # QSS selection/focus/progress
    "canvas_bg": "#0b0f1a",  # node canvas + 2D viewer backgrounds
}
"""Live view of the active theme palette (refreshed by ui.theming)."""

COPYRIGHT_LINE = (
    "\u00a9 2026 Taritolay, Nicol\u00e1s Daniel \u00b7 GPL-3.0-or-later"
)

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)

SPLASH_SIZE = (680, 440)
SPLASH_LOCKUP_WIDTH = 560

# Minimum time the splash stays visible (seconds): startup is fast on
# warm caches, and without a floor the splash just flickers.
SPLASH_MIN_SECONDS = 2.5

# Progress bar geometry (logical pixels, bottom strip of the splash).
SPLASH_BAR = {"x": 60, "y": -52, "width": 560, "height": 4}


def tint_lockup(secondary: QColor) -> QImage:
    """Recolor the lockup PNG preserving its two-tone design.

    Bright pixels (emblem/LYNCEUS) take ``secondary``; mid pixels (SCAN)
    take it darkened, mirroring today's white/gray. Alpha untouched.
    Vectorized with numpy: milliseconds at splash size. Used by the
    theme-aware splash (background = theme surface, lockup = theme text).
    """
    import numpy as np
    from PIL import Image

    lockup = Image.open(SPLASH_LOCKUP_PNG).convert("RGBA")
    data = np.asarray(lockup).astype(np.float32)
    alpha = data[..., 3:4]
    # Luminance over the transparent background-agnostic RGB.
    lum = (
        0.2126 * data[..., 0] + 0.7152 * data[..., 1] + 0.0722 * data[..., 2]
    ) / 255.0
    main = np.array(secondary.getRgb()[:3], dtype=np.float32)
    dark = np.array(secondary.darker(140).getRgb()[:3], dtype=np.float32)
    out = np.zeros_like(data)
    # SCAN gray sits at ~0.70, emblem/LYNCEUS at 1.0: 0.85 splits cleanly.
    bright = lum > 0.85
    mid = (lum > 0.2) & ~bright
    out[..., :3][bright] = main
    out[..., :3][mid] = dark
    out[..., 3:4] = alpha
    tinted = Image.fromarray(out.astype(np.uint8), "RGBA")
    buffer = tinted.tobytes("raw", "RGBA")
    image = QImage(
        buffer, tinted.width, tinted.height, QImage.Format.Format_RGBA8888
    ).copy()
    return image

# Render SVGs at 2x with smoothing: the Inkscape masters are large and the
# runtime sizes are small, so plain scaling leaves a seam on the emblem's
# mirror axis. HiDPI-correct via devicePixelRatio.
_SVG_UPSCALE = 2.0


def _smooth_hints(painter: QPainter) -> None:
    painter.setRenderHints(
        QPainter.RenderHint.Antialiasing
        | QPainter.RenderHint.SmoothPixmapTransform
        | QPainter.RenderHint.TextAntialiasing
    )


def render_svg(path: str, width: int, height: int | None = None) -> QPixmap:
    """Render an SVG asset to a transparent pixmap of exact pixel size.

    Missing/unreadable assets yield a null pixmap; callers fall back
    gracefully (no icon, no splash art) instead of crashing startup.
    """
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return QPixmap()
    default = renderer.defaultSize()
    if height is None:
        height = (
            max(1, round(width * default.height() / default.width()))
            if default.width() > 0
            else width
        )
    image = QImage(
        round(width * _SVG_UPSCALE),
        round(height * _SVG_UPSCALE),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    _smooth_hints(painter)
    # No painter.scale(): render() maps the viewBox onto the whole device,
    # so a 2x device already yields 2x pixels (pre-scaling crops instead).
    renderer.render(painter)
    painter.end()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(_SVG_UPSCALE)
    return pixmap


def tint_pixmap(pixmap: QPixmap, color: QColor) -> QPixmap:
    """Retint white artwork with `color` (single tone via SourceIn).

    Preserves alpha and the HiDPI ratio. Used for the window icon and the
    menubar logo so both follow the theme text color.
    """
    image = pixmap.toImage().convertToFormat(
        QImage.Format.Format_ARGB32_Premultiplied
    )
    painter = QPainter(image)
    painter.setCompositionMode(
        QPainter.CompositionMode.CompositionMode_SourceIn
    )
    painter.fillRect(image.rect(), color)
    painter.end()
    out = QPixmap.fromImage(image)
    out.setDevicePixelRatio(_SVG_UPSCALE)
    return out


def app_icon(tint: QColor | None = None) -> QIcon:
    """Window icon for the whole application (all windows inherit it).

    The master artwork is white-on-transparent, invisible on light title
    bars: pass the theme text color to retint it (same idea as the splash
    lockup, single tone via SourceIn). ``None`` keeps the raw artwork.
    """
    icon = QIcon()
    for size in ICON_SIZES:
        pixmap = render_svg(ICON_SQUARE_SVG, size)
        if pixmap.isNull():
            continue
        if tint is not None:
            pixmap = tint_pixmap(pixmap, tint)
        icon.addPixmap(pixmap)
    return icon


def _compose_splash(bg: str, lockup: QImage) -> QPixmap:
    """Shared layout: lockup image + version, bottom strip left empty."""
    muted = QColor(THEME_COLORS["muted"])
    width, height = SPLASH_SIZE
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(bg))
    painter = QPainter(pixmap)
    _smooth_hints(painter)

    y = 34
    if not lockup.isNull():
        target_w = SPLASH_LOCKUP_WIDTH
        target_h = max(1, round(target_w * lockup.height() / lockup.width()))
        scaled = lockup.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage((width - scaled.width()) // 2, y, scaled)
        y += scaled.height() + 10

    version_font = QFont()
    version_font.setPointSize(scaled_point_size(11))
    painter.setFont(version_font)
    painter.setPen(muted)
    painter.drawText(
        0, y, width, 20, Qt.AlignmentFlag.AlignHCenter, f"v{APP_VERSION}"
    )
    painter.end()
    return pixmap


def splash_base_pixmap(
    bg: str | None = None,
    lockup_path: str | None = None,
) -> QPixmap:
    """Static splash artwork: lockup PNG + version on the theme background.

    The lockup is an Inkscape PNG export (the SVG's live `<text>` does not
    render under Qt); it is retinted with the theme text color via
    `tint_lockup` so the splash follows the active theme, and only the
    version (from ``lynceus.__version__``) is drawn by Qt. The bottom strip
    stays empty for the progress milestones. Missing lockup file degrades
    to version-only instead of crashing.
    """
    lockup_image = QImage(lockup_path or SPLASH_LOCKUP_PNG)
    if not lockup_image.isNull() and lockup_path is None:
        lockup_image = tint_lockup(QColor(THEME_COLORS["text"]))
    return _compose_splash(
        bg or THEME_COLORS["splash_bg"], lockup_image
    )


def splash_pixmap(
    bg: str | None = None,
    logo_path: str | None = None,
) -> QPixmap:
    """Back-compat alias: the static splash artwork (no progress)."""
    return splash_base_pixmap(bg=bg, lockup_path=logo_path)


def set_splash_progress(
    splash: QSplashScreen, base: QPixmap, fraction: float, message: str
) -> None:
    """Advance the splash to an honest milestone: bar + status line.

    The bar only moves on real startup steps (never animated): callers pass
    the fraction matching the work just completed. Recomposes from ``base``,
    swaps the pixmap in and pumps the loop so the step is actually seen
    before the next blocking call.
    """
    from PySide6.QtWidgets import QApplication

    fraction = min(1.0, max(0.0, fraction))
    frame = QPixmap(base)
    width, height = SPLASH_SIZE
    bar = SPLASH_BAR
    bar_y = height + bar["y"] if bar["y"] < 0 else bar["y"]
    painter = QPainter(frame)
    _smooth_hints(painter)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(THEME_COLORS["border"]))
    painter.drawRoundedRect(bar["x"], bar_y, bar["width"], bar["height"], 2, 2)
    if fraction > 0:
        painter.setBrush(QColor(THEME_COLORS["accent"]))
        painter.drawRoundedRect(
            bar["x"], bar_y, max(8, round(bar["width"] * fraction)),
            bar["height"], 2, 2,
        )
    message_font = QFont()
    message_font.setPointSize(scaled_point_size(10))
    painter.setFont(message_font)
    painter.setPen(QColor(THEME_COLORS["muted"]))
    painter.drawText(
        0, bar_y - 28, width, 20, Qt.AlignmentFlag.AlignHCenter, message
    )
    painter.end()
    splash.setPixmap(frame)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
