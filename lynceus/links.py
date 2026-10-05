# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Public web endpoints (domain, no Qt).

Single source of truth for every outbound link: the About page, the Help
menu and (later) the extensions dialog's online-catalog button all read
from here, so a future move (custom domain, new org) touches one file.
The documentation / extensions-catalog paths do not exist yet — the URLs
below are the final ones and start working as soon as the site is live.
"""

WEBSITE_URL = "https://danico19827.github.io/LynceusScan-Web/"
DOCS_URL = WEBSITE_URL + "docs/"
EXTENSIONS_CATALOG_URL = WEBSITE_URL + "extensions/"
CORE_REPO_URL = "https://github.com/Danico19827/LynceusScan"
ISSUES_URL = CORE_REPO_URL + "/issues"
