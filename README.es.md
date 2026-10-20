# LynceusScan — beta pública (0.1.0b5)

> **English version:** [README.md](README.md)

Herramienta visual de flujos de trabajo para datos geoespaciales — LiDAR
primero.

LynceusScan es un editor visual de flujos de trabajo: conecta pasos de
procesamiento en un lienzo y ejecútalos en paralelo sobre un levantamiento
dividido en cuadrículas, con vistas previas 2D/3D de cada producto. Incluye
un núcleo LiDAR (carga, limpieza, clasificación, terreno, métricas, exports)
y acepta extensiones — nodos extra, packs de idioma y temas — como archivos
individuales importables.

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

## Novedades de la beta 5

La beta 5 es la entrega de escala y confianza: el motor ahora termina
datasets profesionales de punta a punta en máquinas normales, y cuando algo
no entra lo dice en vez de bajar la calidad en silencio.

- **Datasets grandes, mismos resultados** — los mosaicos, las derivadas de
  terreno y las uniones de tablas se calculan por partes que caben en tu
  memoria, con el mismo resultado que hacerlo todo de una vez.
- **Mosaicos sin costuras** — DTM, DSM, intensidad, penetración del dosel y
  cobertura de líneas de vuelo conservan entera la celda de borde
  compartida por cuadrículas vecinas; el resultado coincide con una pasada
  global única a 0,3/0,5/1 m, sin líneas de un píxel.
- **Corridas que aguantan datos reales** — las cuadrículas ralas vecinas se
  fusionan en archivos más grandes (hasta ~19× menos archivos y ~4× más
  rápida la división en cuadrículas en nuestro benchmark de miles de
  millones de puntos), las sesiones enormes se dividen en segmentos
  independientes y los mosaicos virtuales unen segmentos sin duplicar
  datos.
- **Fallos honestos** — el motivo de cada nodo fallido queda en la sesión y
  un informe persistente lista los nodos fallidos, su causa y la carpeta de
  la sesión cuando la corrida termina con errores.
- **Asesor pre-vuelo** — estima la RAM desde las cabeceras LAS/LAZ antes de
  correr y ofrece arreglos de un clic (conservar intermedios solo por esta
  corrida, más memoria, fusionar cuadrículas ralas, dividir en segmentos);
  la carpeta de salida es configurable.
- **Colas de archivos** — encola varios archivos en cualquier entrada y
  córrelos desatendidos, una sesión limpia por archivo, con un resumen
  OK/fallidos al final.
- **Línea de comandos (CLI)** — corre proyectos desde un script, de forma
  reproducible, con reporte legible por máquina y códigos de salida (`run`,
  `validate`, `nodes list` incluidos).
- **Nodos nuevos** — Fusionar rásters (ediciones DTM/DSM/CHM, una leyenda
  compartida) y Filtrar por banderas QA (quitar o conservar retornos de
  solape y sintéticos).
- **Operaciones en el lienzo** — multiselección, copiar/pegar/duplicar
  grupos de nodos, deshacer/rehacer y "Asignar archivos..." por lote.
- **Interfaz** — librería de nodos con búsqueda, panel de extensiones
  reconstruido, sesiones de una línea en el panel de ejecuciones, galería
  de productos que se llena durante la corrida, botones de diálogo en los
  nueve idiomas, carpeta de salida configurable y una estética renovada
  (barra de herramientas flotante, cabeceras de nodo con color).
- **Mejoras del visor** — los rásters usan la misma rampa y rango que QGIS
  (con leyenda de gradiente), se respeta el NODATA externo, el zoom es más
  suave y los puntos 3D respetan la profundidad (la grilla de referencia ya
  no se dibuja sobre la nube).
- **Plantillas que viajan** — al abrir una plantilla copiada de otra máquina
  se conservan los archivos que existen y se listan una sola vez los que
  hay que volver a elegir.

## Diseñado con IA, supervisado en varias dimensiones

Este sistema fue diseñado con la ayuda de inteligencia artificial, y
supervisado en varias dimensiones para garantizar su buen
funcionamiento: reglas del dominio LiDAR (ASPRS/USGS), arquitectura del
motor y corrección de datos, traducciones de la interfaz (paridad total
en 9 idiomas), entrega del instalador (verificada lanzando la app
empaquetada) y una auditoría carpeta por carpeta de todo lo que se
distribuye. El `/docs`
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

Abrir un proyecto directo, o correrlo headless:

```powershell
.venv\Scripts\python main.py ruta\al\proyecto.lynx
.venv\Scripts\python -m lynceus.cli run ruta\al\proyecto.lynx   # headless
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
- **Nodos extra** — cinco nodos forestales experimentales (enfoque basado
  en área, claros del dosel, estratos de altura, ápices y copas) y consultas
  meteorológicas Open-Meteo; mismo flujo de importación, aterrizan en la
  librería bajo su propia categoría.

**Plantillas de proyecto** incluidas (`templates/`, Archivo → Nuevo
desde plantilla): ocho flujos listos en español — siete de núcleo
(inspección de archivo, limpieza, terreno, índices de vegetación, datasets
grandes segmentados, comparación de cambio entre dos CHM y diagnóstico de
vuelo) más una cadena de inventario forestal que requiere los packs de
extensión forestry.

## Alcance y límites (lee esto antes de confiar en los productos)

Sin magia silenciosa: cada algoritmo documenta lo que puede y lo que no.

- **Clasificar suelo** trae cinco métodos publicados, cada uno con un
  alcance conocido. PMF conserva por diseño objetos más anchos que su
  ventana máxima; SMRF es por diseño para terreno llano a suave y pela
  el relieve empinado; CSF necesita que sus iteraciones cubran el
  relieve del tile (los defaults cubren 50 m) y no tiene post-proceso
  por pendiente; MCC es el mejor bajo dosel denso sin retornos de
  suelo; ATIN densifica un TIN adaptativo desde semillas de suelo y
  sigue relieve mixto (su espaciado de semillas debe exceder el
  edificio más grande). Fuera de su alcance un método degrada con aviso
  — nunca finge certeza.
- **La validación es con fixtures**: escenas sintéticas analíticas con
  respuesta conocida más un benchmark Tipo I/II, todo en la suite
  abierta de tests. Eso prueba fidelidad a los papers publicados, no
  exactitud sobre tu levantamiento. La exactitud de campo es tarea QA
  del operador — verifica contra puntos de control antes del uso
  operativo, exactamente como exigen las normas ASPRS/USGS.

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
- `lynceus/cli.py` — interfaz de línea de comandos (`run`, `validate`, `nodes list`).
- `lynceus/nodes/` — nodos de procesamiento incluidos (LiDAR, Raster, Table, Flow).
- `lynceus/processing/` — motor de cuadrículas, motor de ejecución, sesiones, proveniencia.
- `lynceus/ui/` — lienzo, inspector, visores, diálogos.
- `lynceus/plugins/` — extensiones (descubrimiento, importador, locales, términos).
- `lynceus/tools/` — CLI del vendor (`extension_tool`: scaffold + pack).
- `templates/` — plantillas de proyecto incluidas.
- `tests/` — suite de la biblioteca estándar (`python -m unittest discover -s tests`).

## Licencia

GPL-3.0-or-later (ver `LICENSE`). Las extensiones cargan in-process y
heredan la GPL. Las contribuciones de terceros se rigen por `CLA.md`.
Si usas este software en investigación, por favor cítalo (ver
`CITATION.cff`).
