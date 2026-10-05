# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""App-level Terms & Conditions (domain, no Qt).

The Terms gate runs on first launch, before the main window is shown. The
user must agree to TERMS_VERSION once per installation; the acceptance is
persisted by ConsentStore in terms.json (sha256 fingerprint of the combined
ES+EN texts + timestamp). Any change to either text re-triggers the gate,
and any new version bumps the record key, re-asking again. The dialog shows
the text matching the active language (``terms_text``), falling back to
English.

This is a legal consent record, not a copy-protection mechanism: the core is
open source under GPL-3.0-or-later. Third-party extensions as derivative works
must comply with GPL. Contributors retain authorship and control of their own
work: the author does not relicense a third-party contribution to commercial
terms without written consent, and the author retains the right to relicense
their own extensions under commercial terms (dual licence). See CLA.md.

Liability notes: the core nodes were developed with AI assistance and their
outputs must be verified by a qualified professional before operational use;
products carry embedded provenance metadata that must be preserved on
redistribution (a GPL-3.0 section 7(b)-(c) style term); users keep the rights
to their fully original creations and hold the author harmless for damage
arising from their own use.
"""

TERMS_ID = "app"
TERMS_VERSION = "v5"

TERMS_TEXT_ES = (
    "LynceusScan — Términos y Condiciones.\n\n"
    "Este software es software libre: su núcleo (core) se distribuye bajo la "
    "Licencia Pública General de GNU, versión 3 o posterior "
    "(GPL-3.0-or-later). Usted puede copiarlo, modificarlo y redistribuirlo "
    "de acuerdo con los términos de esa licencia.\n\n"
    "Extensiones de terceros que se integren a LynceusScan como obras "
    "derivadas del núcleo GPL deben distribuirse bajo GPL-3.0-or-later o "
    "una licencia compatible.\n\n"
    "Contribuciones de terceros: quien contribuye conserva la autoría de su "
    "trabajo y su crédito en el material derivado. El autor no relicenciará "
    "una contribución de terceros a términos comerciales sin el "
    "consentimiento escrito de su autor; dicha relicencia se negocia por "
    "cada contribución. El autor del proyecto conserva el derecho a "
    "relicenciar sus propias extensiones bajo términos comerciales "
    "separados (licencia dual). Consulte CLA.md y LICENSE-COMMERCIAL.md "
    "para más información.\n\n"
    "Metadatos de procedencia: los productos que genera LynceusScan llevan "
    "metadatos de procedencia y autoría incrustados (etiqueta "
    "lynceus_provenance en GeoTIFF, cabecera y VLR en LAS/LAZ, tablas "
    "gpkg_metadata en GeoPackage, sidecar <archivo>.meta.json). Al "
    "redistribuir esos productos, usted debe conservar dichos metadatos y no "
    "tergiversar su origen o autoría. Esta exigencia corresponde a los "
    "términos adicionales permitidos por la GPL-3.0, sección 7(b)-(c), y no "
    "limita los derechos que la GPL otorga sobre el software.\n\n"
    "Asistencia de inteligencia artificial: los nodos del núcleo fueron "
    "desarrollados con asistencia de inteligencia artificial. Los productos "
    "y métricas que genera el software son estimaciones técnicas: deben ser "
    "analizados y verificados por un profesional calificado antes de "
    "cualquier uso operativo, legal o comercial (por ejemplo, valoración "
    "forestal). El autor no responde por decisiones tomadas sobre "
    "resultados no verificados.\n\n"
    "Sus creaciones: usted conserva los derechos sobre lo que cree con el "
    "software siempre que sea obra totalmente original suya (sus proyectos, "
    "sus nodos escritos desde cero, su configuración). Si su creación "
    "modifica o deriva de una obra existente —del autor o de otro creador—, "
    "se aplican las condiciones de esa obra (GPL-3.0-or-later, CLA o "
    "licencia comercial, según corresponda) y debe conservarse la autoría y "
    "atribución originales.\n\n"
    "Identidad del operador: si configura un nombre u organización en "
    "Preferencias → Operador, esa identidad se incrusta en los metadatos de "
    "procedencia de los productos que genere; usted es responsable de su "
    "veracidad.\n\n"
    "Responsabilidad: usted usa el software bajo su propia responsabilidad "
    "y mantendrá indemne al autor frente a reclamos, daños o gastos "
    "derivados de su uso, sus modificaciones, sus redistribuciones o las "
    "decisiones tomadas con los productos generados. En ningún caso el "
    "autor responderá por daños indirectos, incidentales o consecuentes, ni "
    "por el uso indebido del software.\n\n"
    "El software se entrega \"as is\", sin garantía alguna, expresa o "
    "implícita, incluida —sin limitación— la garantía de idoneidad para un "
    "fin particular. El autor no será responsable por daños derivados del "
    "uso del software.\n\n"
    "Estos términos son un registro legal del consentimiento y no restringen "
    "los derechos que la GPL otorga sobre el software. Instalar o usar el "
    "software implica la aceptación de estos términos; la aceptación queda "
    "registrada localmente con la versión y la fecha."
)

TERMS_TEXT_EN = (
    "LynceusScan — Terms & Conditions.\n\n"
    "This software is free software: its core is distributed under the GNU "
    "General Public License, version 3 or later (GPL-3.0-or-later). You may "
    "copy, modify and redistribute it under the terms of that license.\n\n"
    "Third-party extensions integrated into LynceusScan as derivative works "
    "of the GPL core must be distributed under GPL-3.0-or-later or a "
    "compatible license.\n\n"
    "Third-party contributions: contributors retain authorship of their work "
    "and credit in derived material. The author will not relicence a "
    "third-party contribution under commercial terms without its author's "
    "written consent; such relicensing is negotiated per contribution. The "
    "project author retains the right to relicence their own extensions "
    "under separate commercial terms (dual licence). See CLA.md and "
    "LICENSE-COMMERCIAL.md for details.\n\n"
    "Provenance metadata: products generated by LynceusScan carry embedded "
    "provenance and authorship metadata (the lynceus_provenance tag in "
    "GeoTIFF, header and VLR records in LAS/LAZ, gpkg_metadata tables in "
    "GeoPackage, <file>.meta.json sidecars). When redistributing those "
    "products, you must preserve that metadata and must not misrepresent "
    "their origin or authorship. This requirement mirrors the additional "
    "terms permitted by GPL-3.0, section 7(b)-(c), and does not limit the "
    "rights the GPL grants over the software.\n\n"
    "AI assistance: the core nodes were developed with AI assistance. The "
    "products and metrics produced by the software are technical estimates: "
    "they must be reviewed and verified by a qualified professional before "
    "any operational, legal or commercial use (for example, timber "
    "valuation). The author is not liable for decisions made on unverified "
    "outputs.\n\n"
    "Your creations: you retain the rights to what you create with the "
    "software as long as it is your fully original work (your projects, "
    "nodes written from scratch, your configuration). If your creation "
    "modifies or derives from an existing work — by the author or another "
    "creator — that work's terms apply (GPL-3.0-or-later, CLA or commercial "
    "licence, as applicable) and the original authorship and attribution "
    "must be preserved.\n\n"
    "Operator identity: if you configure a name or organization in "
    "Preferences → Operator, that identity is embedded in the provenance "
    "metadata of the products you generate; you are responsible for its "
    "accuracy.\n\n"
    "Liability: you use the software at your own risk and shall hold the "
    "author harmless against claims, damages or expenses arising from your "
    "use, your modifications, your redistributions or decisions made with "
    "generated products. In no event shall the author be liable for "
    "indirect, incidental or consequential damages, or for misuse of the "
    "software.\n\n"
    "The software is provided \"as is\", without warranty of any kind, "
    "express or implied, including — without limitation — the warranty of "
    "fitness for a particular purpose. The author shall not be liable for "
    "damages arising from the use of the software.\n\n"
    "These terms are a legal record of consent and do not restrict the "
    "rights the GPL grants over the software. Installing or using the "
    "software implies acceptance of these terms; acceptance is recorded "
    "locally with the version and the date."
)

# Backwards-compatible alias: the Spanish text (v4 and earlier were ES-only).
TERMS_TEXT = TERMS_TEXT_ES

TERMS_TEXTS = {"es": TERMS_TEXT_ES, "en": TERMS_TEXT_EN}


def terms_text(lang: str | None = None) -> str:
    """Terms text for a language code (exact match, else English)."""
    return TERMS_TEXTS.get((lang or "").lower(), TERMS_TEXT_EN)


def terms_blob() -> str:
    """Combined ES+EN texts: the single fingerprint covers both languages."""
    return TERMS_TEXT_ES + "\n\x00\n" + TERMS_TEXT_EN
