# LynceusScan — beta pública (0.1.0b3)

> **English version:** [README.md](README.md)

ETL visual agnóstico para datos geoespaciales — LiDAR primero.

LynceusScan es un editor de pipelines por nodos: conecta nodos de
procesamiento en un lienzo y ejecútalos sobre datos teselados en
paralelo, con vistas previas 2D/3D de cada producto. Incluye un núcleo
LiDAR (carga, limpieza, clasificación, terreno, métricas raster,
exports) y acepta extensiones — nodos extra, packs de idioma y temas —
como archivos individuales importables.

- Sitio y docs: https://danico19827.github.io/LynceusScan-Web/
- Extensiones: https://github.com/Danico19827/LynceusScan-Extensions
- Issues: https://github.com/Danico19827/LynceusScan/issues
- Contacto: lynceusscan@gmail.com

## Beta pública

Esto es una **beta pública**: corre flujos LiDAR reales de punta a
punta, pero espera cambios rompientes — contratos de nodos, layouts de
sesión y archivos de proyecto todavía pueden evolucionar antes de la
1.0. Reportes de errores, sugerencias de mejora y nuevas extensiones
son bienvenidos (ver Contribuir abajo).

## Diseñado con IA, supervisado en varias dimensiones

Este sistema fue diseñado con la ayuda de inteligencia artificial, y
supervisado en varias dimensiones para garantizar su buen
funcionamiento: reglas del dominio LiDAR (ASPRS/USGS), arquitectura del
motor y corrección de datos, traducciones de la interfaz (paridad total
en 4 idiomas), entrega del instalador (verificada con smoke test) y una
auditoría carpeta por carpeta de todo lo que se distribuye. El `/docs`
del repositorio de desarrollo guarda el informe normativo del dominio;
donde este README sea escueto, el código es el contrato.

## Instalación

**Instalador Windows** (sin terminal): descarga el setup desde
[Releases](https://github.com/Danico19827/LynceusScan/releases). Se
instala por usuario (sin permisos de admin), asocia archivos de proyecto
`.lynx` y conserva tus datos al desinstalar por defecto. Las builds beta
no están firmadas, así que Windows SmartScreen pedirá confirmación en
la primera ejecución.

**Desde fuentes** — requisitos: Windows 10/11, Python 3.13.

```powershell
git clone https://github.com/Danico19827/LynceusScan.git
cd LynceusScan
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

Abrir un proyecto directo:

```powershell
.venv\Scripts\python main.py ruta\al\proyecto.lynx
```

## Qué incluye (y qué no)

El núcleo corre en inglés e incluye el set de nodos LiDAR, el motor de
procesamiento, visores 2D/3D, plantillas de proyecto, 22 temas de
interfaz y el framework de extensiones. **No** incluido, por diseño —
descárgalo del [repositorio de
extensiones](https://github.com/Danico19827/LynceusScan-Extensions):

- **Packs de idioma** (español, ruso, italiano, portugués brasileño, chino simplificado, hindi, árabe, francés, bengalí) —
  importa el `.lxpkg` con Archivo → Extensiones → Importar extensión (o
  arrastra el archivo al lienzo). El idioma nuevo aparece en
  Herramientas → Preferencias → General.
- **Nodos extra** (métricas forestales, consultas meteorológicas, ...) —
  mismo flujo de importación; aterrizan en la librería bajo su propia
  categoría.

**Plantillas de proyecto** incluidas (`templates/`, Archivo → Nuevo
desde plantilla): un workflow de modelo de altura de dosel y un vuelo
multiespectral SfM con NDVI — ambas solo-núcleo, listas para adaptar a
tus propios archivos.

## Hazlo tuyo: escribe una extensión

LynceusScan está hecho para personalizarse: un nodo nuevo es un solo
archivo `.py`, un pack de idioma o tema es una carpeta con
`manifest.json`. La [guía de autoría](EXTENSIONES.es.md) ([in
English](EXTENSIONS.md)) te lleva de cero a un `.lxpkg` instalable sin
adivinanzas — sin SDK, sin compilador, sin registro. Si adaptas la
herramienta a tu propio flujo, considera compartir la extensión para que
otros también la usen.

## Contribuir

- **Reportes de errores**: abre un issue con pasos para reproducir, ideal
  con un proyecto `.lynx` mínimo.
- **Sugerencias de mejora**: abre un issue describiendo qué flujo de
  trabajo desbloquearía el cambio.
- **Nuevas extensiones**: constrúyela con la guía de autoría y
  compártela vía el repositorio de extensiones.
- Las contribuciones de código se rigen por `CLA.md`.

## Estructura

- `main.py` — punto de entrada de la aplicación.
- `lynceus/nodes/` — nodos de procesamiento built-in (LiDAR, Raster, Table, Flow).
- `lynceus/processing/` — motor de teselado, ejecutor DAG, sesiones, proveniencia.
- `lynceus/ui/` — lienzo, inspector, visores, diálogos.
- `lynceus/plugins/` — extensiones (discovery AST, importador, locales, términos).
- `lynceus/tools/` — CLI del vendor (`extension_tool`: scaffold + pack).
- `templates/` — plantillas de proyecto incluidas.
- `tests/` — suite de la biblioteca estándar (`python -m unittest discover -s tests`).

## Licencia

GPL-3.0-or-later (ver `LICENSE`). Las extensiones cargan in-process y
heredan la GPL. Las contribuciones de terceros se rigen por `CLA.md`.
Si usas este software en investigación, por favor cítalo (ver
`CITATION.cff`).
