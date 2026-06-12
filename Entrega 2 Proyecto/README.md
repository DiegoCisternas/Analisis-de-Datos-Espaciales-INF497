# Análisis espacial de reportes SOSAFE — INF-497

Análisis de datos espaciales sobre reportes ciudadanos de seguridad en cinco comunas de la RM de Santiago: **Santiago, Ñuñoa, Las Condes, Maipú y Puente Alto**.

## Datos requeridos (no incluidos en esta entrega)

**Esta entrega no incluye datos**: los reportes SOSAFE son confidenciales (nos fueron confiados como grupo) y el resto se deriva de ellos. Para ejecutar el proyecto se debe contar con estos archivos, con estos nombres exactos, en la raíz del proyecto:

| Archivo / carpeta | Contenido | Cómo obtenerlo |
|---|---|---|
| `reports/` | Parquet diarios crudos de SOSAFE (`AAAA-MM-DD.parquet`, nov 2023 – mar 2025) | **Entregados por la profesora Daniela Opitz**; solicitar acceso a ella |
| `dataset_pois.csv` | 973 POIs de Google Places (botillerías, discotecas, estaciones de metro, comisarías) | Generar con `google_maps_api.py` (requiere API key propia de Google Maps) |
| `comunas.geojson` | Límites oficiales de las 5 comunas (INE) | Lo genera/cachea `scripts/pipeline_sosafe.py` en la primera ejecución |
| `dataset_analitico.parquet` | Reportes SOSAFE limpios y enriquecidos (876.449 filas) | `python scripts/pipeline_sosafe.py` |
| `grilla_h3_res8.gpkg` | Grilla hexagonal H3 res 8 con variables agregadas por celda | `python scripts/grilla_h3.py` |
| `dataset_pois_enriquecido.csv` | POIs con conteos de reportes/delitos en 100/200/500 m | `python pois_reportes.py` |

## Fuentes de datos

- **SOSAFE**: los datos crudos de la aplicación (reportes ciudadanos georreferenciados, con descripción, código de tipo, timestamp y geometría) **nos fueron entregados por la profesora Daniela Opitz**. No se redistribuyen.
- **Google Maps**: los POIs fueron extraídos **usando la API de Google Maps** (Places, *Nearby Search*) con [google_maps_api.py](google_maps_api.py): grid search de 5 nodos por comuna (radio 2 km) para 4 categorías, con eliminación de duplicados por el traslape de radios.

Para solicitar los datos revisar este link de drive:
https://drive.google.com/drive/folders/1ydCmskWzTU7XEO0_wLxosKYDU3NTavF5?usp=drive_link

## Qué se hizo con los datos

**`dataset_analitico.parquet`** — salida de [scripts/pipeline_sosafe.py](scripts/pipeline_sosafe.py): (1) carga de los parquet diarios; (2) parseo de geometría WKB y limpieza (nulos, GPS fuera de la RM, duplicados); (3) asignación de comuna por point-in-polygon; (4) clasificación del reporte y variable objetivo `es_delito` (códigos de tipo ∪ keywords en la descripción); (5) unión espacial con `dataset_pois.csv` en UTM 19S: distancia al POI más cercano por categoría y conteos en 200/500 m; (6) variables temporales (período del día, nocturno, fin de semana, mes). Es el insumo de la grilla H3, el ESDA y los análisis temporales.

**`dataset_pois_enriquecido.csv`** — generado por [pois_reportes.py](pois_reportes.py): la vista inversa, cuántos reportes/delitos ocurren alrededor de cada POI.

## Estructura

- [00_proyecto_completo.ipynb](00_proyecto_completo.ipynb): notebook unificado (procesamiento, grilla H3, ESDA Moran/LISA, ESDA temporal, EDA de delitos). La lógica vive en los módulos de [scripts/](scripts/).

## Antes de usar el notebook

El notebook no genera los datasets: los lee. Por eso, con `reports/` y `dataset_pois.csv` ya en esta carpeta (ver tabla de arriba), hay que correr los scripts **en este orden**:

```bash
# 1. Obligatorio — crudos SOSAFE -> dataset_analitico.parquet (también cachea comunas.geojson)
python scripts/pipeline_sosafe.py

# 2. Obligatorio — dataset analítico -> grilla hexagonal H3 (grilla_h3_res8.gpkg)
python scripts/grilla_h3.py

# 3. Opcional — POIs enriquecidos (solo para el análisis de POIs conflictivos)
python pois_reportes.py
```

Tras los pasos 1 y 2 existe todo lo que el notebook necesita; se ejecuta de arriba hacia abajo con el kernel en esta misma carpeta. El resto del análisis (Moran, LISA, espacio-temporal, EDA) lo llama el propio notebook desde los módulos de `scripts/`.

> Dependencias: `pandas`, `geopandas`, `h3`, `libpysal`, `esda`, `splot`, `fastparquet`, `scipy`, `matplotlib`, `seaborn`.
