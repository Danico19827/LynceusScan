# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Plugin system: extensions (node packs) discovery and loading. The domain
layer is Qt-free; the UI lives under lynceus/ui/.

An extension/artifact is a pack: manifest.json + node .py files (+ optional
EULA text). The core is open source under GPL-3.0-or-later; third-party packs
as derivative works inherit GPL. The author retains the right to relicense
their own extensions under commercial terms, but does not relicense a
third-party contribution to commercial terms without the contributor's
written consent, and always preserves the contributor's attribution
(see terms.py, CLA.md and LICENSE-COMMERCIAL.md). There is no licensing
machinery: no signed blobs, no accounts, no gates beyond EULA acceptance
consent.
"""
