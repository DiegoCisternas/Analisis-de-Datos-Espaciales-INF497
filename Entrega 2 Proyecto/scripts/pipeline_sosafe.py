"""
Pipeline reproducible de procesamiento espacial - INF-497 Entrega 2.

Flujo crudo -> analitico:
  1. Carga de reportes SOSAFE (parquet diarios).
  2. Parseo de geometria (WKB) y limpieza (nulos, bounding box RM, duplicados).
  3. Poligonos comunales (Nominatim, cacheados en comunas.geojson).
  4. Asignacion de comuna (point-in-polygon) y recorte a la zona de estudio.
  5. Clasificacion de reportes: categoria inferida + variable objetivo es_delito
     (type nucleo {0,1,2,6} OR keywords de delito en la descripcion).
  6. Reproyeccion a UTM 19S (EPSG:32719) y union con POIs de Google Places:
     distancia al POI mas cercano por categoria + conteos en radios (200/500 m).
  7. Variables derivadas temporales (periodo, nocturno, fin de semana, mes).
  8. Guardado del dataset analitico (dataset_analitico.parquet).

Uso:  python pipeline_sosafe.py
"""
import glob, os, time, unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
RUTA_REPORTS = "reports"
RUTA_POIS = "dataset_pois.csv"
CACHE_COMUNAS = "comunas.geojson"
SALIDA = "dataset_analitico.parquet"

COMUNAS = ["Santiago", "Puente Alto", "Ñuñoa", "Maipú", "Las Condes"]
CRS_GEO = "EPSG:4326"       # lat/lon
CRS_METRICO = "EPSG:32719"  # UTM 19S (metros) para distancias en Santiago

# Limites comunales oficiales (INE, Censo 2017, Region Metropolitana / R13).
RUTA_SHP_COMUNAS = os.path.join("..", "github", "geodata", "datos", "external",
                                "censo2017", "R13", "COMUNA_C17.shp")
# codigo CUT de las 5 comunas de estudio -> etiqueta usada en el proyecto
CODIGOS_COMUNA = {"13101": "Santiago", "13119": "Maipú", "13120": "Ñuñoa",
                  "13114": "Las Condes", "13201": "Puente Alto"}

# Bounding box amplio de la Region Metropolitana (descarta GPS erroneos)
BBOX_RM = dict(lon_min=-71.2, lon_max=-70.3, lat_min=-33.85, lat_max=-33.15)

# type nucleo que representa delito de forma directa (inferido de las descripciones)
TYPES_DELITO = {0, 1, 2, 6, 28, 112, 139, 146}

# categorias donde el keyword genera falsos positivos claros (no se marca por keyword):
#   mascotas       -> "se robaron a mi gata", "perro perdido posiblemente robado"
#   mal_estacionado-> autos cuyo texto menciona robo/droga de contexto, no el evento
CATS_SIN_KEYWORD = {"mascotas", "mal_estacionado"}

# Mapeo de codigo `type` -> categoria legible (inferido por muestreo de descripciones).
# El `type` de SOSAFE es ruidoso: los codigos 17, 31, 33, 124 son cajones de sastre
# (mezclan corte de luz + spam de marketplace + ruidos + balazos) y quedan en "otro".
MAPA_TYPE = {
    # --- delito / inseguridad ---
    0: "asalto", 146: "asalto",
    1: "robo", 112: "robo", 139: "robo",
    2: "robo_domicilio", 6: "sospechoso", 28: "vandalismo",
    118: "violencia_intrafamiliar", 48: "violencia_intrafamiliar",
    # --- emergencias ---
    11: "accidente_transito", 18: "emergencia_salud",
    19: "bomberos_incendio", 49: "quema_humo",
    # --- convivencia / orden publico ---
    13: "ruidos_molestos", 91: "ruidos_molestos", 5: "ruidos_molestos",
    14: "disturbios", 23: "comercio_ambulante",
    43: "situacion_calle", 68: "situacion_calle",
    # --- busquedas ---
    32: "mascotas", 56: "mascotas", 67: "mascotas", 79: "mascotas",
    147: "persona_extraviada",
    # --- servicios / infraestructura (consolidado) ---
    22: "electricidad", 101: "electricidad", 136: "electricidad",
    152: "electricidad", 133: "electricidad", 141: "electricidad", 58: "electricidad",
    26: "infra_vial", 24: "infra_vial", 38: "infra_vial",
    41: "infra_vial", 50: "infra_vial",
    42: "agua_alcantarillado", 29: "agua_alcantarillado",
    20: "arboles", 21: "basura", 140: "basura",
    # --- transito ---
    39: "mal_estacionado", 103: "mal_estacionado",
    # --- ruido de datos (no son eventos reales) ---
    16: "prueba_alarma",
    70: "marketplace", 72: "marketplace", 46: "marketplace",
    53: "marketplace", 76: "marketplace",
    # --- mixto irreductible ---
    17: "otro", 31: "otro", 33: "otro", 124: "otro", 150: "otro", 96: "otro",
    97: "otro",  # "SAFE Tag encontrado" (meta de la app, n=4 en toda la RM)
}

# keywords de delito (sobre texto normalizado SIN tildes / minuscula).
# El sufijo \w* captura conjugaciones/plurales (robar, robaron, robando, ...).
# Se evitan raices ambiguas (p.ej. "bala" -> balance, "tiro" -> tirado, "choro" -> chorizo).
KW_DELITO = (
    r"\b(?:"
    # --- robo / hurto ---
    r"robo|roban|robaron|robando|robar|hurto|hurtaron|ladron|arrebat|"
    r"mechero|carterist|reducidor|monre|alunizaje|"
    # --- asalto / encerrona ---
    r"asalt|encerrona|portonazo|lanzazo|turbazo|motochorro|cogote|atraco|atracador|"
    # --- delincuente / jerga ---
    r"delincuen|antisocial|lacra|flaite|maleant|encapuchado|pandill|"
    # --- armas / disparos ---
    r"balacera|balazo|balea|disparo|tiroteo|arma de fuego|arma blanca|"
    r"pistola|revolver|escopeta|navaja|punal|municion|"  # "armado/armados" excluido: FP con "arbol armado"
    # --- violencia grave ---
    r"amenaz|apunal|acuchill|asesin|homicid|mataron|secuestr|raptaron|"
    r"violacion|abuso sexual|"
    # --- drogas ---
    r"droga|narco|microtrafico|traficante|pasta base|"
    # --- danos a la propiedad ---
    r"vandaliz|vandalico|saqueo|saquearon|saquean"
    r")\w*"
)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def quita_tildes(serie: pd.Series) -> pd.Series:
    """Normaliza a minuscula sin tildes para matching robusto de keywords."""
    return (serie.fillna("")
            .str.normalize("NFKD")
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
            .str.lower())


# --------------------------------------------------------------------------- #
# 1. Carga
# --------------------------------------------------------------------------- #
def cargar_reportes(ruta=RUTA_REPORTS) -> pd.DataFrame:
    archivos = sorted(glob.glob(os.path.join(ruta, "*.parquet")))
    dfs = [pd.read_parquet(f, engine="fastparquet") for f in archivos]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[1] Carga: {len(archivos)} archivos -> {len(df):,} reportes crudos")
    return df


# --------------------------------------------------------------------------- #
# 2. Geometria + limpieza
# --------------------------------------------------------------------------- #
def limpiar(df: pd.DataFrame) -> gpd.GeoDataFrame:
    n0 = len(df)
    df = df[df["geometry"].notna()].copy()
    geom = gpd.GeoSeries.from_wkb(df["geometry"], crs=CRS_GEO)
    gdf = gpd.GeoDataFrame(df.drop(columns="geometry"), geometry=geom, crs=CRS_GEO)

    # filtro bounding box RM (descarta coordenadas fuera de rango)
    gdf = gdf[(gdf.geometry.x.between(BBOX_RM["lon_min"], BBOX_RM["lon_max"])) &
              (gdf.geometry.y.between(BBOX_RM["lat_min"], BBOX_RM["lat_max"]))]
    n_bbox = len(gdf)

    # deduplicacion: mismo texto + mismo instante + mismo tipo = reporte repetido
    gdf["_lon"] = gdf.geometry.x.round(6)
    gdf["_lat"] = gdf.geometry.y.round(6)
    gdf = gdf.drop_duplicates(subset=["created_at", "description", "type", "_lon", "_lat"])
    gdf = gdf.drop(columns=["_lon", "_lat"])

    print(f"[2] Limpieza: {n0:,} -> sin geom nula/bbox {n_bbox:,} -> sin duplicados {len(gdf):,}")
    return gdf.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Poligonos comunales (cache + Nominatim)
# --------------------------------------------------------------------------- #
def obtener_comunas(shp=RUTA_SHP_COMUNAS, cache=CACHE_COMUNAS) -> gpd.GeoDataFrame:
    """Limites oficiales INE (Censo 2017). Filtra las 5 comunas, reproyecta a 4326
    y cachea a GeoJSON para portabilidad."""
    if os.path.exists(shp):
        g = gpd.read_file(shp)
        g["COMUNA"] = g["COMUNA"].astype(str)
        g = g[g["COMUNA"].isin(CODIGOS_COMUNA)].copy()
        g["comuna"] = g["COMUNA"].map(CODIGOS_COMUNA)
        gdf = g[["comuna", "COMUNA", "geometry"]].to_crs(CRS_GEO).reset_index(drop=True)
        gdf.to_file(cache, driver="GeoJSON")
        print(f"[3] Comunas: shapefile INE Censo 2017 ({len(gdf)} comunas) -> cache {cache}")
        return gdf
    if os.path.exists(cache):
        gdf = gpd.read_file(cache)
        print(f"[3] Comunas: cache {cache} ({len(gdf)} comunas)  [shapefile no encontrado]")
        return gdf
    raise FileNotFoundError(f"No se encontro el shapefile de comunas ({shp}) ni el cache ({cache}).")


# --------------------------------------------------------------------------- #
# 4. Asignar comuna y recortar
# --------------------------------------------------------------------------- #
def asignar_comuna(gdf, gdf_com) -> gpd.GeoDataFrame:
    gdf_com = gdf_com[["comuna", "geometry"]].to_crs(CRS_GEO)
    res = gpd.sjoin(gdf, gdf_com, how="inner", predicate="within").drop(columns="index_right")
    print(f"[4] Recorte a las 5 comunas: {len(gdf):,} -> {len(res):,} dentro de la zona de estudio")
    print("    " + res["comuna"].value_counts().to_string().replace("\n", "\n    "))
    return res.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Clasificacion (variable objetivo)
# --------------------------------------------------------------------------- #
def clasificar(gdf) -> gpd.GeoDataFrame:
    gdf["categoria_inferida"] = gdf["type"].map(MAPA_TYPE).fillna("otro")
    texto = quita_tildes(gdf["description"])
    kw_match = texto.str.contains(KW_DELITO, regex=True, na=False)
    # no marcar por keyword en categorias con FP claros (ver CATS_SIN_KEYWORD)
    kw_match = kw_match & ~gdf["categoria_inferida"].isin(CATS_SIN_KEYWORD)
    es_type = gdf["type"].isin(TYPES_DELITO)
    gdf["match_keyword"] = kw_match
    gdf["es_delito"] = (es_type | kw_match)
    print(f"[5] Clasificacion: es_delito = {gdf['es_delito'].sum():,} "
          f"({gdf['es_delito'].mean()*100:.1f}%)  | por type={es_type.sum():,} "
          f"| por keyword={kw_match.sum():,}")
    return gdf


# --------------------------------------------------------------------------- #
# 6. Union con POIs (distancias + conteos en radio)
# --------------------------------------------------------------------------- #
def unir_pois(gdf, ruta_pois=RUTA_POIS, radios=(200, 500)) -> gpd.GeoDataFrame:
    df_p = pd.read_csv(ruta_pois)
    gpois = gpd.GeoDataFrame(df_p,
                             geometry=gpd.points_from_xy(df_p.Longitud, df_p.Latitud),
                             crs=CRS_GEO).to_crs(CRS_METRICO)
    gdf_m = gdf.to_crs(CRS_METRICO)
    xy = np.c_[gdf_m.geometry.x.values, gdf_m.geometry.y.values]

    for cat in sorted(gpois["Categoria"].unique()):
        sub = gpois[gpois["Categoria"] == cat]
        pts = np.c_[sub.geometry.x.values, sub.geometry.y.values]
        tree = cKDTree(pts)
        dist, _ = tree.query(xy, k=1)
        col = cat.replace(" ", "_")
        gdf[f"dist_{col}_m"] = np.round(dist, 1)
        for r in radios:
            gdf[f"n_{col}_{r}m"] = tree.query_ball_point(xy, r, return_length=True)
    print(f"[6] Union POIs: distancia + conteos (radios {radios} m) por categoria "
          f"{sorted(gpois['Categoria'].unique())}")
    return gdf


# --------------------------------------------------------------------------- #
# 7. Variables derivadas temporales
# --------------------------------------------------------------------------- #
def derivar_temporales(gdf) -> gpd.GeoDataFrame:
    ca = gdf["created_at"]
    gdf["fecha"] = ca.dt.strftime("%Y-%m-%d")
    gdf["anio"] = ca.dt.year
    gdf["mes"] = ca.dt.month
    gdf["fin_de_semana"] = gdf["dow"].isin([5, 6])             # dow: lunes=0 ... domingo=6
    bins = [-1, 5, 11, 18, 23]
    labels = ["madrugada", "manana", "tarde", "noche"]
    gdf["periodo"] = pd.cut(gdf["hour"], bins=bins, labels=labels)
    gdf["nocturno"] = (gdf["hour"] >= 20) | (gdf["hour"] <= 5)  # ventana asociada a alcohol/ocio
    print("[7] Variables derivadas: fecha, anio, mes, fin_de_semana, periodo, nocturno")
    return gdf


# --------------------------------------------------------------------------- #
# 8. Guardado
# --------------------------------------------------------------------------- #
def guardar(gdf, salida=SALIDA):
    out = gdf.copy()
    out["lon"] = out.geometry.x
    out["lat"] = out.geometry.y
    out["geometry_wkt"] = out.geometry.to_wkt()
    out["periodo"] = out["periodo"].astype(str)  # categorical -> str para parquet
    out = pd.DataFrame(out.drop(columns="geometry"))
    out.to_parquet(salida, engine="fastparquet", index=False)
    print(f"[8] Guardado: {salida}  ({len(out):,} filas, {out.shape[1]} columnas)")
    return out


def main():
    t0 = time.time()
    df = cargar_reportes()
    gdf = limpiar(df)
    com = obtener_comunas()
    gdf = asignar_comuna(gdf, com)
    gdf = clasificar(gdf)
    gdf = unir_pois(gdf)
    gdf = derivar_temporales(gdf)
    out = guardar(gdf)
    print(f"\nLISTO en {time.time()-t0:.0f}s. Columnas finales:\n{list(out.columns)}")


if __name__ == "__main__":
    main()
