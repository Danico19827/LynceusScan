# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""QSettings keys shared across the UI (no Qt)."""

SETTINGS_ORG = "LynceusScan"
SETTINGS_APP = "LynceusScan"
OPERATOR_NAME_KEY = "operator_name"
OPERATOR_ORG_KEY = "operator_org"
RECENT_KEY = "recent_projects"
MAX_RECENT = 10
SHOW_START_KEY = "show_start_dialog"
# Preferences (Tools > Preferences...): performance + general.
CPU_PERCENT_KEY = "cpu_percent"
CPU_PERCENT_DEFAULT = 100
MEMORY_PERCENT_KEY = "memory_percent"
MEMORY_PERCENT_DEFAULT = 25
ACCEL_MODE_KEY = "acceleration_mode"
ACCEL_BACKEND_KEY = "acceleration_backend"
ACCEL_DEVICE_KEY = "acceleration_device"
# Preferences round 2: display budgets, canvas behavior, operator.
DISPLAY_BUDGET_KEY = "display_point_budget"
DISPLAY_BUDGET_DEFAULT = "auto"
DISPLAY_RASTER_CELLS_KEY = "display_raster_cells"
DISPLAY_RASTER_CELLS_DEFAULT = 4_000_000
DISPLAY_TABLE_ROWS_KEY = "display_table_rows"
DISPLAY_TABLE_ROWS_DEFAULT = 5000
RECENT_LIMIT_KEY = "recent_limit"
RECENT_LIMIT_DEFAULT = 10
SNAP_KEY = "snap_enabled"
# Finals-only sessions: keep per-tile intermediates + shared tile cache.
KEEP_INTERMEDIATES_KEY = "keep_intermediates"
# Color theme pack id (Preferences > General); "default" is the core dark.
THEME_KEY = "theme"
THEME_DEFAULT = "default"
# Interface font family (Preferences > General); "" is the system default.
FONT_FAMILY_KEY = "font_family"
FONT_FAMILY_DEFAULT = ""
# Interface font size as percent (Preferences > General); 100 = theme size.
FONT_SIZE_KEY = "font_size"
FONT_SIZE_DEFAULT = 100