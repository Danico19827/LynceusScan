# Guía de autoría de extensiones de LynceusScan (provisional)

> **English version:** [EXTENSIONS.md](EXTENSIONS.md)

> **Estado:** guía de autoría provisional. Es completa para construir
> extensiones funcionales sin adivinanzas, pero escueta: un manual
> oficial pulido con estándares de compatibilidad viene después. Ante
> duda, el código es el contrato — cada sección cita el archivo fuente
> que la impone.

> **Beta pública:** LynceusScan está en beta pública, así que contratos
> de nodos, layouts de sesión y archivos de proyecto todavía pueden
> evolucionar antes de la 1.0. Construye contra los contratos de esta
> guía (metadata literal `NODE_*`, puertos tipados, `PROCESSING_SPECS`)
> y tu extensión acompañará al núcleo. Este sistema fue diseñado con la
> ayuda de inteligencia artificial y supervisado en varias dimensiones
> (reglas del dominio LiDAR, arquitectura del motor, paridad de
> traducciones, entrega del instalador, auditoría de lo distribuido) —
> si una sección abajo discrepa del archivo fuente citado, el archivo
> gana y merece un bug report. Las nuevas extensiones son bienvenidas:
> adapta la herramienta a tu flujo y compártelas por el [repositorio de
> extensiones](https://github.com/Danico19827/LynceusScan-Extensions)
> para que otros también las usen.

## 0. Qué es una extensión

LynceusScan procesa LiDAR (y afines) como un grafo de **nodos**: cajas
en un lienzo cableadas por puertos tipados. Una **extensión** agrega
capacidades sin tocar el núcleo, en una de estas clases:

| Clase | Qué es | Unidad de distribución |
|---|---|---|
| `node` (autocontenido) | Un archivo `.py`: un nodo de procesamiento, una entrada de archivo, o un nodo con puertos custom | Archivo `.py`, o bundle `.lxpkg` |
| `locale` (pack) | Un catálogo de traducción que habilita un idioma de UI | Carpeta con `manifest.json`, o `.lxpkg` |
| `theme` (pack) | Una paleta de 12 colores que re-estiliza toda la app | Carpeta con `manifest.json`, o `.lxpkg` |
| `font` (futuro) | Todavía no es clase de pack. Las fuentes de interfaz se eligen de familias del sistema en Preferencias → General; packs con TTF se enchufarán al mismo selector después (`lynceus/ui/fonts.py`) | — |

**Reglas base (sin excepciones):**
- Las extensiones cargan **in-process** y son obra derivada: heredan
  **GPL-3.0-or-later**. Marca la metadata `license` en consecuencia (si
  la omites aplica GPL-3.0-or-later por defecto).
- No hay **maquinaria de licencias**: sin archivos `.lic`, cuentas,
  firmas ni pagos. El único gate es una **EULA** opcional por nodo
  (texto de consentimiento mostrado una vez antes del primer uso).
- Los archivos de nodo se leen por **AST primero, se ejecutan solo al
  correr**. Todo lo que la app necesite sin ejecutar tu código (ids,
  puertos, schemas, specs) **debe ser un literal plano** a nivel de
  módulo — sin valores computados, sin llamadas a funciones, sin imports
  necesarios para leerlos.

## 1. Qué necesitas

- Un checkout fuente de LynceusScan para correr el CLI de autoría
  (`python -m lynceus.tools.extension_tool ...` desde la raíz del repo).
  Sin él igual puedes escribir a mano cada archivo de abajo (son texto
  plano/JSON) y dejar que el diálogo de importación los revise al entrar.
- Un editor de texto. Sin SDK, sin compilador, sin registro en ningún lado.

Comandos CLI (desde la raíz del repo):

```
python -m lynceus.tools.extension_tool scaffold <dir> --kind node|locale|theme --author "Your Name"
python -m lynceus.tools.extension_tool pack <files...> -o out.lxpkg
python -m lynceus.tools.extension_tool validate <path>
```

## 1b. Dependencias garantizadas (qué puede importar tu código)

Dentro de tareas tile/barrier (workers incluidos, app instalada
incluida) puedes contar con la **biblioteca estándar** más exactamente
estos paquetes pinneados:

| Import | Paquete | Úsalo para |
|---|---|---|
| `numpy` | numpy | arrays, histogramas, matemática vectorizada |
| `laspy` | laspy | nubes de puntos LAS/LAZ |
| `rasterio` | rasterio | lectura/escritura GeoTIFF, transforms, CRS |
| `scipy` | scipy | KD-trees, filtros, stats |
| `shapely` | shapely | geometrías |
| `geopandas` | geopandas | tablas vectoriales (lento; prefiere `sqlite3` para lecturas GPKG) |
| `PIL` | Pillow | encode/decode/resize de imágenes |
| `psutil` | psutil | introspección de memoria/procesos |
| `PySide6` | PySide6 | Qt (rara vez lo necesitan los nodos; los widgets sí) |
| `OpenGL` | PyOpenGL | GL crudo (internos del visor 3D) |

Todo lo demás **no está garantizado** (puede existir en tu venv de dev
por accidente vía otro paquete y desaparecer en la app instalada). O
evítalo, o vendorea el archivo junto a tu nodo, o falla fuerte con un
mensaje legible:

```python
try:
    import cv2
except ImportError:
    raise RuntimeError(
        "Image Filter needs OpenCV, which is not bundled: "
        "pip install opencv-python-headless or vendor cv2 next to this file"
    )
```

## 2. Tu primer nodo en 10 minutos (nodo de procesamiento barrier)

Un nodo **barrier** corre una vez por corrida (sin tiling): lee
entradas, escribe productos, devuelve un payload. Copia este skeleton —
está completo y corre:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Jane Doe <jane@example.com>
"""Node - Uppercase CSV (stack example)."""

from lynceus.nodes.ports import PortType

NODE_ID = "acme.uppercase"
NODE_NAME = "Uppercase CSV"
NODE_CATEGORY = "Table"
NODE_AUTHOR = "Jane Doe"
NODE_FOLDER = "acme"
NODE_LICENSE = "GPL-3.0-or-later"

INPUTS = (PortType.TABLE_CSV,)
OUTPUTS = (PortType.TABLE_CSV,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_uppercase",
    "input_ports": ("table_csv",),
    "output_files": {"table_csv": "upper.csv"},
    "output_port": "table_csv",
    "output_globs": ("upper.csv",),
    "config_schema": {
        "column": {
            "type": "str",
            "default": "name",
            "description": "Column to uppercase.",
            "impact": "Only this column is rewritten; the rest passes through.",
            "group": "Options",
        },
    },
}


def get_config_defaults():
    return {"column": "name"}


def barrier_uppercase(ctx):
    import csv
    from pathlib import Path

    from lynceus.nodes._paths import scoped_file
    from lynceus.processing import provenance

    src = Path(ctx.get("table_csv_path", ""))
    if not src.is_file():
        raise RuntimeError("Uppercase CSV needs a table on its input")
    column = ctx.get("column", "name")
    out_path = Path(scoped_file(ctx, "upper.csv"))
    with open(src, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if column in row and row[column]:
            row[column] = row[column].upper()
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    prov_doc = provenance.build_provenance(ctx, "upper.csv")
    provenance.write_sidecar(out_path, prov_doc, {"column": column})
    return {
        "file": str(out_path),
        "kind": "Table",
        "node": NODE_ID,
        "rows": len(rows),
    }
```

Después:

```
python -m lynceus.tools.extension_tool validate acme_uppercase.py
python -m lynceus.tools.extension_tool pack acme_uppercase.py -o uppercase.lxpkg
```

Importa `uppercase.lxpkg` vía Archivo → Extensiones → *Importar
extensión...* (o drag & drop). Aterriza en `extensions/nodes/acme/`,
aparece en la librería bajo Table, se cablea a cualquier salida de
tabla, y su CSV abre en el visor de tablas de la galería.

## 3. Referencia de metadata del nodo

Literales a nivel de módulo (leídos por AST;
`lynceus/plugins/registry.py`):

| Constante | Requerida | Significado |
|---|---|---|
| `NODE_ID` | **sí** | Id único estable, convención `<carpeta>.<nodo>`. Renombrarlo huérfana proyectos guardados. |
| `NODE_NAME` | **sí** | Etiqueta visible (traducible, §9). |
| `NODE_AUTHOR` | **sí** | Atribución. Import rechaza archivos sin él. Debe ser un nombre real — placeholders (`Test`, `todo`, `xxx`, `your name`…) fallan revisión y `validate` avisa. |
| `NODE_CATEGORY` | no | Nivel superior de librería (`"LiDAR"`, `"Raster"`, `"Table"`, `"Flow"`, o la tuya). |
| `NODE_SUBCATEGORY` | no | Segundo nivel de librería. |
| `NODE_DESCRIPTION` | no | HTML mostrado en la librería (traducible, conserva el marcado `<b>/<i>/<code>`). |
| `NODE_FOLDER` | no | Subcarpeta de instalación; por defecto el prefijo de `NODE_ID`. |
| `NODE_LICENSE` | no | Id SPDX; omitido ⇒ `GPL-3.0-or-later`. |
| `NODE_EULA` | no | Texto de consentimiento mostrado una vez antes del primer uso (queda en inglés). |
| `NODE_DISCLAIMER` | no | Texto corto estampado en la proveniencia de cada producto (§6). Queda en inglés. |
| `NODE_HOMEPAGE` / `NODE_REPOSITORY` / `NODE_ISSUES` / `NODE_DOCUMENTATION` / `NODE_DONATE` / `NODE_COPYRIGHT` / `NODE_CONTRIBUTORS` / `NODE_TAGS` / `NODE_CHANGELOG` | no | Metadata blanda de confianza/atribución mostrada en el diálogo de Extensiones. |
| `NODE_TRANSLATIONS` | no | Dict `{"lang": {"origen": "traducción"}}` de traducciones fallback (§9). |
| `INPUTS` / `OUTPUTS` | — | Tuplas de `PortType`, strings de id custom, `(PortType, "etiqueta")` o `PortDef(...)` (§4). `INPUTS = ()` significa fuente. |
| `PORT_TYPE_DEFS` | no | Tipos de puerto custom (§4). |
| `PROCESSING_SPECS` | — | Dict de abajo; solo valores literales son visibles a la UI. |
| `get_config_defaults()` | no | Devuelve `{clave: default}`; los defaults del motor salen del schema mismo. |

## 4. Puertos

Los ids de puerto builtin viven en el enum `PortType`
(`lynceus/nodes/ports.py`): `point_cloud`, `tiles`, `dtm_tile`,
`dsm_tile`, `dtm_mosaic`, `dsm_mosaic`, `chm_mosaic`, `grid_metrics`,
`grid_vegetation`, `canopy_penetration`, `coverage_raster`,
`intensity_ortho`, `strata_raster`, `vector`, `table_csv`, `table_gpkg`,
`table`, `raster` (más `classified_point_cloud` por compatibilidad).

Declarar un extremo:

```python
from lynceus.nodes.ports import PortDef, PortType

INPUTS = (
    PortType.CHM_MOSAIC,                            # entrada requerida
    PortDef(PortType.POINT_CLOUD, required=False),  # opcional: silencio si suelta
    PortDef(PortType.TABLE, name="Table A"),        # etiquetado (traducible)
    PortDef(PortType.TABLE, name="Table B", group="tables"),  # grupo exclusivo
)
```

- `required=False` hace el puerto opcional (un cable faltante es
  silencio, nunca error). Entradas requeridas bloquean la corrida con
  mensaje legible.
- `group="..."` declara exclusión mutua: solo un puerto del grupo puede
  cablearse a la vez (el canvas rechaza el segundo cable).
- Dos entradas del **mismo** tipo llegan posicionalmente como
  `ctx["in0_path"]`, `ctx["in1_path"]`, ... Una entrada única de un tipo
  llega como `ctx["<port_type>_path"]` (p. ej. `ctx["table_csv_path"]`).
  Los ids custom conservan sus puntos verbatim: el puerto `demo.image`
  llega como `ctx["demo.image_path"]`. Cuando varios proveedores
  podrían matchear, gana el primero — lee la clave, con `in0_path` como
  fallback posicional.

Tipos de puerto custom (sin cambios al núcleo):

```python
PORT_TYPE_DEFS = [
    {"id": "acme.image", "display_name": "Image", "color": "#7dd3a8",
     "compatible": ("acme.image", "raster"), "viewer_kind": "raster"},
]
```

- `id`: string único. `display_name`: etiqueta en inglés (clave
  traducible, §9). `color`: color `#rrggbb` de cable/plug.
  `compatible`: tupla de ids de entrada que esta salida puede alimentar
  (ids builtin o del mismo archivo).
- `viewer_kind` (opcional): reusa un visor existente — `raster`
  (GeoTIFFs con CRS), `image` (PNG/JPG/BMP/GIF/WEBP planos, sin
  georreferencia), `vector`, `table_csv`, `point_cloud`, `json`. Hoy
  **no hay forma** de agregar una clase de visor nueva desde una
  extensión; elige la existente más cercana (un PNG con `viewer_kind:
  "raster"` previsualiza como raster).
- Ids desconocidos degradan con gracia (gris `#888888`, sin conexiones).
- `validate` chequea forma de entradas, formato `#rrggbb`, y avisa sobre
  ids `compatible` que no matchean builtins ni el mismo archivo (typos).
- Declara cada id custom **una vez** entre tus archivos: el registro es
  last-wins en orden de discovery, así que un duplicado *conflictivo*
  recablea en silencio (`validate` lo falla entre packs y bundles, y
  avisa en repeticiones idénticas). Productor + consumidor compartiendo
  un tipo es el caso normal — duplica solo el acuerdo, nunca copias
  divergentes. Validar un archivo solo avisa sobre ids declarados en sus
  hermanos; ese aviso desaparece al validar carpeta, pack o bundle
  juntos.

## 5. Contratos de procesamiento

### 5a. Tareas barrier (empieza acá — cubre inputs, combinadores, exports)

Firma: `def barrier_<nombre>(ctx: dict) -> dict`. Corre **una vez por
corrida**, después de todo el trabajo tile del que depende. Úsala para
todo lo que no sea trabajo paralelo por tile.

Claves de `ctx` en las que puedes contar:

| Clave | Significado |
|---|---|
| `session_dir` | Dir absoluto de artifacts de la sesión. **Escribe solo bajo él** (vía `scoped_file`, nunca `session/<nombre>` hardcodeado). |
| `node_iid` | Id de esta instancia (scoping de ramas, subcarpetas por instancia). |
| `module_id` | Tu `NODE_ID`. |
| `file_path` | Archivo elegido en un widget file-input (§7), o `""`. |
| `<port_type>_path` | Salida del proveedor para una entrada de tipo único. |
| `in0_path`, `in1_path`, ... | Salidas de proveedores para entradas repetidas del mismo tipo, en orden de declaración. |
| `<config_key>` | Valores de tu `config_schema`, planos (defaults del schema pre-aplicados). |
| `tile_scope` | String de scope de rama; pásalo por `scoped_file`, nunca a fingerprints. |

Payload de retorno (contrato de galería + reuso):

```python
{
    "file": str(out_path),   # DEBE existir: archivos anunciados-pero-ausentes
                              # son error fuerte, nunca fantasma silencioso.
    "kind": "Table",         # etiqueta humana (agrupado de galería).
    "node": NODE_ID,
    "warnings": [...],       # opcional; se muestran ámbar, quedan en inglés.
    "display_name": ...,     # etiqueta opcional de hoja; default basename.
}
```

- Claves string extra `file_2`, `csv`, ... apuntando a archivos
  existentes se vuelven productos extra de galería. `node`/`kind`/
  `source_file`/`warnings`/`count` y amigas son metadata, nunca productos.
- Errores: lanza `RuntimeError("...")` con mensaje en inglés plano (la UI
  lo muestra sobre el nodo). Todo lo que el usuario pueda rodear con
  config va a `warnings` — los warnings **quedan en inglés por diseño**.
- Determinismo es propiedad de corrección: mismos inputs + misma config
  deben producir bytes idénticos (sin timestamps, sin azar,
  estructuras ordenadas). Reuso, proveniencia y galería lo asumen.
  (Formatos comprimidos pueden diferir byte a byte entre OS/zlib — el
  reuso clavea fingerprints, nunca comparación de bytes — pero mantén
  tus salidas deterministas igual.)

Dónde escribir (esta regla es load-bearing — errarla rompe aguas abajo
en silencio):

- **Barrier de procesamiento/combinador** (final compartido estilo
  mosaico): declara `output_files: {"<puerto>": "<basename>"}` y escribe
  exactamente ese basename vía `scoped_file(ctx, "<basename>")`.
  Aguas abajo resuelve `session/<basename>` (o el dir de scope de rama).
- **Barrier file-input** (producto por instancia, §7): declara
  `session_file: "<basename>"` (nombre fijo, convierte formatos cuando
  toque) y escribe exactamente
  `Path(ctx["session_dir"]) / ctx["node_iid"] / "<basename>"`.
  Aguas abajo resuelve `session/<src_iid>/<basename>`, así dos inputs
  de la misma clase conviven. `scoped_file` está **mal acá**: escribe la
  raíz de sesión mientras el motor mira dentro de la carpeta de
  instancia, y el consumidor recibe un path colgando.
- Nunca hardcodees paths `session/<nombre>` a mano en ningún caso; nunca
  escribas fuera de `session_dir`.

### 5b. Tareas tile (trabajo paralelo por tile)

Firma: `def tile_<nombre>(tile: dict, ctx: dict) -> dict`. Solo para
trabajo que particiona espacialmente (nubes de puntos, rasters). Si tu
algoritmo lee archivos enteros y escribe un producto, usa barrier (§5a)
— el tiling es por archivo LiDAR; fuentes teseladas custom deben
resolver a archivos point-cloud que el motor pueda teselar.

Declara `"tile_task": "tile_<nombre>"` en `PROCESSING_SPECS` y lee
entradas tile igual (paths de `ctx`). Parciales mergean en tu
`barrier_task` o un `consolidator`.

### 5c. `config_schema` (formulario del Inspector)

```python
"res_m": {
    "type": "float",        # str | int | float | bool | str+options
    "default": 0.5,         # debe matchear el tipo; fuente de defaults del motor
    "minimum": 0.05, "maximum": 100.0,   # solo numéricos (int/float)
    "options": [("Label", "value"), ...],# o "enum": [...]; pares label/data
    "description": "...",   # qué hace (traducible, §9)
    "impact": "...",        # cuándo subir/bajar + efectos (traducible)
    "group": "Grid",        # header libre de sección (traducible)
    "advanced": True,       # oculta la línea inline de impacto (el tooltip queda)
},
```

- Tipos: `str` (también fallback para tipos faltantes/desconocidos),
  `int` (±2³¹), `float` (3 decimales en UI — nunca cuentes con más
  precisión editada), `bool`, select vía `options`/`enum` (los data
  deben ser `str`/`int`/`float`).
- Labels renderizan como `key.replace("_", " ")`; `description` +
  `impact` van a tooltips. Los grupos son libres; reusa nombres
  hermanos (`Grid`, `Output`, `Advanced`, `Behavior`...).
- Restricciones: valores **JSON-serializables** (`str/int/float/bool` y
  listas planas) — los fingerprints hacen `json.dumps` de la config sin
  fallback. La clave `strategy` está reservada para familias estrategia
  (§8).

### 5d. Reuso (por qué importan `output_files` / `output_globs`)

```python
"output_files": {"table_csv": "upper.csv"},  # {out_port_id: basename}
"output_port": "table_csv",
"output_globs": ("upper.csv",),
```

Declaran el layout de sesión para que re-corridas con fingerprints
idénticos hardlinkeen tus productos en vez de recomputar. Reglas: solo
anuncia archivos que escribas de verdad; mantén basenames estables entre
configs; nunca escribas fuera del dir de sesión.

Extensiones variables (`.jpg` hoy, `.png` mañana): NO preserves la
extensión fuente. Normaliza cada corrida a un `session_file` fijo
(convierte el formato cuando toque) y declara globs literales cubriendo
el producto **y** su sidecar, p. ej. `"session_file": "image.png"` más
`output_globs = ("image.png", "image.meta.json")`. Rationale, verificado
contra el motor:
- Sin `output_files`/`session_file`, nodos aguas abajo no reciben **ningún
  path** (el injector salta proveedores que no puede ubicar) —
  `validate` avisa.
- El reuso matchea globs contra basenames reales: un glob que no
  matchea nada (p. ej. `input_image.*` para un producto guardado como
  `photo.jpg`) deshabilita el reuso de ese nodo en silencio, para
  siempre recomputando.
- Los sidecars viajan solo cuando un glob matchea su basename
  (`image.*` cubre `image.meta.json`; si no, lístalo explícito).

Qué cubre el fingerprint (para razonar sobre reuso): el string de id de
módulo, los **bytes del archivo fuente del módulo** (editar tu `.py` —
o el archivo de variante activa — siempre recomputa), tu config de
instancia como JSON (claves ordenadas), fingerprints aguas arriba, el
scope del lote, y la identidad del operador cuando hay. **No** incluye
sidecars: los timestamps `generated_at` difieren legítimo entre
corridas, así que compara bytes de producto para determinismo, nunca
sidecars. Mantén basenames estables y salidas deterministas.

### 5e. Ciclo de vida de config (renombres, bajas, versiones)

- Los proyectos guardados guardan tu config como `{clave: valor}`
  plano. Al restaurar, claves desconocidas las **ignora la UI** (solo
  renderizan claves del schema actual) pero viajan sin daño; una clave
  renombrada cambia el fingerprint, así el nodo recomputa seguro en vez
  de reusar productos rancios.
- Reglas: nunca reuses un nombre de clave dado de baja con distinto
  significado o tipo; mantén defaults tipo-estables; `NODE_VERSION`
  está reservado para migraciones futuras (decláralo hoy si planeas
  cambios rompientes, la maquinaria lo honrará después).
- Familias estrategia: el valor `strategy` viaja dentro de la config,
  así cambiar producto/método siempre recomputa — las variantes nunca
  comparten reuso entre claves.

## 6. Proveniencia y estilos (haz lo estándar)

```python
from lynceus.processing import provenance

prov_doc = provenance.build_provenance(ctx, "upper.csv")
provenance.write_sidecar(out_path, prov_doc, {"column": column})
```

- `build_provenance` tolera claves `ctx` faltantes (cae a constantes de
  sistema; operador solo si configurado). `write_sidecar` mergea en
  `<stem>.meta.json` — el path del producto con su extensión
  **reemplazada** (`Path("image.png").with_suffix(".meta.json")` →
  `image.meta.json`, nunca `image.png.meta.json`) — sin pisar claves del
  nodo.
- GeoTIFF/GPKG/LAZ reciben tags embebidos automáticamente **solo** en
  las ramas core de validación; clases custom conservan el sidecar
  (alcanza para galería + lectura sidecar de QGIS).
- `NODE_DISCLAIMER` (si defines uno) se embebe como `node_disclaimer`
  en cada producto: corto, en inglés, factual.
- Estilos QGIS: `PROCESSING_SPECS["qml"] = {"upper.csv": ...}` — id de
  rampa raster (`"dtm"`, `"chm"`, `"ndvi"`, `"generic"`...), nombre de
  campo vectorial, o `{"field", "classes": [...]}` / `{"name", "stops":
  [...]}` explícito; `{"self_styled": True}` cuando escribes el `.qml`
  tú. Sin datos válidos ⇒ se salta en silencio, nunca error.

## 7. Nodos file-input por convención (sin edits al núcleo)

Cualquier nodo con `input_file_key` en `PROCESSING_SPECS` recibe
automático el item Browse + label de archivo (filtros de diálogo desde
`file_filters`; el path elegido siempre llega como
`ctx["file_path"]`). El barrier reusa el framework compartido con su
**propio** dict target — sin filas de registro. `barrier_input_file`
escribe `session/<node_iid>/<session_file>` internamente, así usarlo
cumple la regla de layout §5a automático; solo barriers a mano deben
replicar ese layout:

```python
PROCESSING_SPECS = {
    "barrier_task": "barrier_input_txt",
    "input_file_key": "file_path",
    "file_filters": "Text files (*.txt)",
    "output_port": "table_csv",
    "session_file": "words.csv",
    "kind_group": "none",     # sin rama core de validación: copia fiel
    "kind_label": "Text",
}

def barrier_input_txt(ctx):
    from lynceus.nodes._product_input import barrier_input_file
    return barrier_input_file(ctx, {
        "node_id": NODE_ID, "node_name": NODE_NAME,
        "output_port": "table_csv", "session_file": "words.csv",
        "kind_label": "Text", "kind_group": "none",
    })
```

- `kind_group` elige validación: `"raster"` (aviso monobanda),
  `"multispectral"` (chequeo multibanda), `"table_gpkg"` (SQLite),
  cualquier otro (rama CSV), `"none"`/`"custom"`/`""` (sin opiniones —
  para clases de producto sin rama core).
- Sin `input_file_key`, un barrier corre headless sobre
  `ctx["file_path"]` (widget genérico, sin Browse). Familias estrategia
  con file inputs conservan registro explícito de widget (edit core) —
  nodos planos nunca lo necesitan.

## 8. Familias estrategia (variantes)

Para un nodo con productos/métodos conmutables (como Input Raster con
`strategy=dtm|dsm|chm` o Classify Ground con `pmf|csf|smrf`):

- **Base**: nodo normal con `NODE_ID`, `INPUTS = ()`, `OUTPUTS = ()`, y
  un campo `strategy` en `config_schema` (default `""`) más
  `get_config_defaults() -> {"strategy": ""}`. Sin estrategia elegida
  el nodo es **inerte** (0 puertos) y corridas cableadas bloquean con
  *"Select a product strategy in node(s): …"*.
- **Variante**: un módulo **sin `NODE_ID`** declarando `VARIANT_OF`
  (= id base), `VARIANT_KEY`, `VARIANT_LABEL`, más sus propios
  `INPUTS`/`OUTPUTS`/`PROCESSING_SPECS`/funciones task. `requires`/
  `provides`/puertos de la variante reemplazan los de la base; solo
  `config_schema` mergea (la base conserva `strategy` para el
  dropdown). El valor `strategy` viaja en el fingerprint, así las
  variantes nunca comparten reuso.
- Limitación (honesta): los widgets selectores Producto/Método/Merge
  están registrados explícito en el núcleo — una familia de terceros
  renderiza con el item genérico salvo que el núcleo la mapee. Inputs
  standalone (§7) no necesitan registro alguno.

## 9. Traducciones (tu nodo multilingüe)

La app traduce por **match exacto de inglés**; claves faltantes caen a
inglés, así traducciones parciales distribuyen seguro. Solo idiomas
habilitados por un pack `locale` instalado se ofrecen — tus
traducciones nunca pueden dejar la app a medio traducir solas.

Qué es traducible en tu archivo (todo leído por AST):

- `NODE_NAME`, `NODE_CATEGORY`, `NODE_SUBCATEGORY`, `NODE_DESCRIPTION`
  (conserva `\n` y el marcado `<b>/<i>/<code>` intactos en
  traducciones).
- `config_schema`: labels (`param_name` con `_` → espacio),
  `description`, `impact`, `group`, y labels de `options` (los data
  quedan).
- Etiquetas de puerto: `PortDef(..., name="Table A")` explícito o tuplas
  `(PortType, "etiqueta")` — y `display_name`s de `PORT_TYPE_DEFS`.
- `VARIANT_LABEL`s.

Provéelas vía sidecar junto a tu archivo (`<nodo>.i18n.json`, viaja en
`.lxpkg` automático, gana sobre todo):

```json
{"es": {"Uppercase CSV": "CSV en mayúsculas", "Table A": "Tabla A"}}
```

o vía dict `NODE_TRANSLATIONS = {"es": {...}}` en el `.py`
(precedencia: catálogo del pack < `NODE_TRANSLATIONS` < sidecar).

Queda en inglés por diseño (nunca traduzcas): `NODE_ID`, warnings,
`NODE_EULA`/`NODE_DISCLAIMER`, y tokens técnicos (`CHM`, `NODATA`,
`LAI`, unidades, `Ctrl+…`, `JSON`/`CSV`…). Traduce sentido, no
palabras: los impactos deben explicar qué hacer y qué se rompe, en
lenguaje natural — una traducción palabra-por-palabra que el usuario no
pueda accionar es peor que el inglés.

## 10. Packs de idioma (nuevos idiomas)

```
my-lang/
  manifest.json   {"schema_version": 2, "id": "fr", "version": "0.1.0",
                   "kind": "locale", "display_name": "...",
                   "author": "...", "payload": {"catalog": "fr.json"}}
  fr.json         {"fr": {"Source English string": "Translation", ...}}
```

- La clave del catálogo es el código de idioma; el **set de claves
  debería espejar el catálogo de referencia** (copia uno y traduce
  valores — `validate` avisa sobre valores vacíos y mismatches de
  placeholders (`{x}`)).
- Importar el pack solo **habilita** el idioma (Preferencias →
  General); nunca lo activa solo. El inglés siempre es fallback.

## 11. Packs de tema

```
my-theme/
  manifest.json   {... "kind": "theme", "payload": {"palette": "theme.json"}}
  theme.json      {"background": "#...", ...}   // 12 roles fijos abajo
```

| Rol | Default | Usado para |
|---|---|---|
| `background` | `#0b0f1a` | Ventana, lienzo, splash |
| `surface` | `#1c2333` | Paneles, menús, relleno de nodo |
| `border` | `#2f3a4f` | Bordes, borde de nodo |
| `border_strong` | `#3a465c` | Hovers, selección, selección de árbol |
| `text` | `#e6e9f0` | Texto primario, títulos, lockup del splash |
| `text_body` | `#aeb7c6` | Texto de cuerpo/labels |
| `muted` | `#8a93a6` | Texto secundario, cables, categorías |
| `accent` | `#7d9fd4` | Selección, foco, progreso, links |
| `danger` | `#e08a8a` | Hover de cerrar, errores |
| `amber` | `#c9a86a` | Valores modificados, warnings |
| `canvas_bg` | `#0b0f1a` | Fondo del lienzo de nodos |
| `wire` | `#8a93a6` | Cables de conexión |

- Cualquier subset vale (roles faltantes caen a defaults); roles
  desconocidos y `#rrggbb` malformados se ignoran con warning y nunca
  tumban el arranque. Colores semánticos (verdes/rojos de estado, chips
  de aviso, viewport 3D) quedan fijos por diseño.
- Importar habilita el tema en Preferencias → General (aplica en vivo);
  nunca lo activa solo.

## 12. Fuentes (setting hoy, packs mañana)

Las fuentes de interfaz se eligen de **familias del sistema** en
Preferencias → General (default = default del sistema); cada `QFont()`
en widgets e items pintados del canvas sigue la fuente de la
aplicación, así un setting re-skinea todo el texto. Solo se ofrecen
faces escalables con cobertura Latin (las bitmap/símbolo/display fallan
o rompen layouts). Nunca hardcodees una familia en un nodo o widget.
Los packs de fuentes (`kind=font` con TTF) registrarán familias en el
mismo selector después — hoy no escribas nada font-específico en
extensiones.

## 13. Confianza, licencia, nodos experimentales

- `NODE_LICENSE`: id SPDX, omitido ⇒ `GPL-3.0-or-later`. Las
  extensiones cargan in-process (obra derivada) y heredan GPL — la ley
  es el límite, no el código.
- `NODE_EULA`: texto de consentimiento mostrado **una vez** antes del
  primer uso (queda en inglés). Úsalo para lo que el usuario deba
  aceptar activamente.
- `NODE_DISCLAIMER`: texto corto **estampado en cada producto** (queda
  en inglés). Forma recomendada para trabajo no verificado/experimental
  (algoritmos asistidos por IA, precisión sin medir): estado
  experimental + requisito de validación profesional + responsabilidad
  del usuario, p. ej. el gate forestal. Nunca prometas precisión que no
  mediste.

## 14. Referencia de tooling

```
# Nuevo skeleton de pack (node|locale|theme; --author requerido)
python -m lynceus.tools.extension_tool scaffold my-pack --kind node --author "Jane Doe"
# Distribuir: un .lxpkg por nodo (.py + sidecar viajan juntos)
python -m lynceus.tools.extension_tool pack mynode.py -o mynode.lxpkg
# Chequear sin instalar (errores fallan, warnings aconsejan)
python -m lynceus.tools.extension_tool validate mynode.py
python -m lynceus.tools.extension_tool validate mynode.lxpkg
python -m lynceus.tools.extension_tool validate my-pack/
```

En app: Archivo → Extensiones → *Importar extensión...* (abre en
filtro Package), *Importar carpeta de pack...*, o drag & drop de
archivos sobre el diálogo. Importa un archivo a la vez: con dos `.py`
(p. ej. un input más su nodo de proceso), importa el primero, después
el segundo — cada uno aterriza junto a sus hermanos por `NODE_FOLDER` y
ambos registran en el próximo discovery. Ids `PORT_TYPE_DEFS`
compartidos entre tus archivos están bien si son idénticos (declara una
vez si puedes; conflictos fallan `validate`). Un `.lxpkg` de pack (con
`manifest.json`) de clase `theme`/`locale` se extrae a
`extensions/themes/<id>/` / `extensions/locales/<id>/` — el mismo
layout que importar la carpeta; packs clase `node` siguen copiados
legacy. Re-importar un pack extraído sin `force` se rechaza;
`force=True` re-extrae encima. Import es idempotente; sobreescribir
contenido distinto se rechaza con error salvo importado con
`force=True` vía API. Por fila *Disable* (queda listado, fuera de
librería/pipeline), *Remove* (borra archivos, pregunta antes), *Open
folder*. Packs deshabilitados nunca registran — deshabilitar el
idioma/tema activo cae a inglés / default.

### 14b. Sin CLI (sin checkout fuente)

Cada artefacto de arriba es escribible a mano; el CLI solo acelera:

- **Nodo**: copia el skeleton §2 en un `.py`. Sin manifest: el diálogo
  de importación acepta un `.py` suelto directo (chequea `NODE_ID` +
  `NODE_AUTHOR` al entrar y reporta errores en message box).
- **`.lxpkg` es un ZIP**: pack = archivos en la raíz del archive. Un
  bundle no trae `manifest.json` (`.py` de nodos + sidecars
  `<nodo>.i18n.json`); un pack completo trae `manifest.json` + payload
  en raíz. Cualquier zip tool sirve — renombra `.zip` a `.lxpkg`.
  Entradas unsafe (`..`, paths absolutos) se rechazan al importar.
- **Locale/theme**: escribe a mano las formas JSON de §10/§11; importa
  la carpeta vía *Importar carpeta de pack...* o zipeala como `.lxpkg`.
- **`validate` no disponible**: confía en los mensajes del diálogo de
  importación (los mismos cheques corren al entrar) más el checklist
  §15.

## 15. Checklist de testing (antes de distribuir)

1. `validate` sobre el `.py` **y** sobre el `.lxpkg` terminado: cero errores.
2. Importa en la app, cablea el nodo, corre: productos en galería con
   el visor correcto; re-corrida reusa (fingerprint estable).
3. Disable/re-enable desde el diálogo; remove y re-import limpio.
4. Si multilingüe: claves del sidecar matchean las fuentes inglesas
   exacto (placeholders intactos).
5. Determinismo: corre dos veces, `diff` a los productos (bytes deben
   matchear; sidecars pueden diferir en `generated_at`, productos no).
6. Verifica el path real en disco de cada producto tras la primera
   corrida (debe estar donde §5a dice que aguas abajo lo buscará).
7. Corre dos veces y compara hashes de sidecars: idénticos salvo timestamps.

## 16. Distribución

- Un `.lxpkg` por nodo (`.py` + sidecar `<nodo>.i18n.json`, sin bundle),
  o carpeta de pack con `manifest.json` para packs locale/theme.
- Versión en `manifest.json` (`version` + `compatibility` con versiones
  de app testeadas); `author` requerido, `homepage`/`repository`/
  `issues`/`documentation`/`donate`/`changelog`/`tags`/`contributors`/
  `copyright` opcionales pero recomendados. Los `NODE_ID`s son para
  siempre: renombrar huérfana proyectos guardados y anula reuso.
- Comparte vía Releases/drive/web: usuarios importan con *Importar
  extensión...*.

## 17. Troubleshooting

| Síntoma | Causa → fix |
|---|---|
| Import dice "no NODE_ID / no NODE_AUTHOR" | Literales faltantes o computados — declara ambos como strings planos. |
| El nodo nunca aparece en la librería | Pack deshabilitado; `INPUTS`/`OUTPUTS` inparseables; archivo sin `NODE_ID` se ignora (helpers). Corre `validate`. |
| Nodo estrategia inerte (0 puertos), corrida bloqueada | Sin estrategia elegida: elige Producto/Método/Merge en el nodo. |
| Error fantasma de barrier (`file` no escrito) | El barrier anunció un producto que nunca escribió — escribe cada archivo anunciado, siempre. |
| Puertos no conectan | Mismatch `compatible` (chequea ids de ambos lados) o conflicto `group` en destino (un cable por grupo exclusivo). |
| Producto sin preview | Extensión de archivo desconocida (la galería rutea: `.tif` raster, imágenes `.png/.jpg/.jpeg/.bmp/.gif/.webp`, `.gpkg` vector, `.csv/.txt` tabla, `.las/.laz` nube, `.json` árbol) o `viewer_kind` faltante en puerto custom. |
| Idioma/tema no ofrecido | Sin pack habilitante instalado (traducciones solas nunca habilitan); pack deshabilitado. |
| Re-corrida recomputa todo | `output_files`/`output_globs` mal o basenames inestables; config no-JSON; bytes no deterministas. |
| Texto UI trabado en inglés | Mismatch de clave con la fuente inglesa (match exacto incl. `\n`/marcado) o idioma sin pack habilitante. |
| `validate` avisa `input_file_key` sin `barrier_task` | El Browse mostraría pero nada corre — cablea el task o saca la clave. |
| Aguas abajo lee paths vacíos/ausentes | El productor omitió `output_files`/`session_file`: no se inyecta path (avisa en `validate`). Normaliza a un `session_file` fijo. |
| Aguas abajo vacío AUNQUE `session_file` declarado | El productor escribió en otro lado: los inputs deben escribir exactamente `session/<own_iid>/<session_file>` (un `scoped_file` suelto aterriza en la raíz de sesión, donde nadie mira). Chequea en disco que el archivo exista en el path inyectado. |
| `validate` falla en port ids conflictivos | Dos archivos declaran el mismo id de puerto distinto — unifica a una definición; last-wins recablearía en silencio. |

## 18. Modelo de ejecución (diseña nodos pesados alrededor de esto)

- **Workers, no la app**: tareas tile y barrier corren en procesos
  worker `spawn` separados (pool al tamaño de CPU), nunca en el proceso
  GUI. Mantén tasks autocontenidas como funciones de `(tile, ctx)` /
  `(ctx)`: **todo lo que toquen debe ser picklable** (dicts/listas/
  strings/números/paths-como-strings planos — sin QObjects, sin file
  handles abiertos, sin lambdas, sin cachés mutables a nivel de módulo
  compartidas con la UI).
- **Envelope de memoria**: el motor throttlea despacho predictivo
  (~70% de RAM sobre tiles en vuelo) y reactivo (muestreo RSS
  achicando workers); el tiler pagina por bucket bajo presupuesto en
  bytes. Diseña para eso igual: streamea archivos (nunca `read()` de un
  raster mayor que RAM — lecturas por ventana llegan post-MVP; hoy,
  rasters oversized fallan el nodo con error legible en vez de swapear
  la máquina), prefiere generadores sobre listas materializadas, cierra
  archivos explícito.
- **Fallas**: cualquier excepción marca el task fallido y cascadea a
  dependientes; el mensaje llega al nodo en el canvas. `RuntimeError`
  con mensaje plano es la convención; `MemoryError`/`KeyError`/
  crashes nativos se comportan igual (fallido, nunca zombi) pero
  testéalos — un segfault mata solo su worker.
- **Cancel/pause**: el motor pollea flags de cancel/pausa entre tasks;
  barriers largos deberían checkpointear a disco periódico para que una
  corrida cancelada retome barato en re-corrida (el reuso clavea
  archivos terminados).
- **Orden de discovery**: `extensions/` camina ordenado y recursivo,
  determinista — último registro gana para port ids duplicados (de ahí
  la regla de definición única §4).
- **Política de rasters grandes (v1)**: el tiling cubre nubes de puntos;
  barriers raster que no quepan en RAM deben fallar rápido con error
  legible nombrando el faltante. Procesamiento por ventanas es roadmap,
  no v1.

## 19. Sandbox y capacidades (la política honesta)

- No hay **sandbox**: las extensiones cargan in-process bajo GPL (obra
  derivada) con los mismos privilegios OS que la app — archivos, red,
  subprocesos. Diseña tu nodo en consecuencia y di lo que hace: si toca
  la red, spawnea procesos, o lee fuera del session dir + archivos
  elegidos, decláralo en `NODE_DESCRIPTION` y gatea lo irreversible
  tras el `NODE_EULA`.
- Nunca escribas fuera de `session_dir` salvo la lectura de la fuente
  elegida; nunca llames a casa en silencio; nunca bloquees el worker en
  input interactivo. Los usuarios otorgan confianza por pack/nodo al
  importar + EULA — mantenlo auditable: archivos chicos, metadata
  literal, sin ofuscación.
