"""
Punto 2 - Unidad de analisis: grilla hexagonal H3 (Uber) resolucion 8.

Construye el dataset analitico AGREGADO POR CELDA a partir de:
  - dataset_analitico.parquet  (reportes SOSAFE enriquecidos, nivel punto)
  - dataset_pois.csv           (POIs Google Places)
  - comunas.geojson            (limites INE para teselar la zona de estudio)

Por que H3 res 8:
  - Area uniforme (~0.74 km2) y vecindad regular (6 vecinos) -> mejor que grilla
    cuadrada para autocorrelacion espacial (sin sesgo de esquinas/MAUP de bordes).
  - 488 celdas con reportes, conteos estables (mediana ~807 reportes/celda) ->
    tasas de delito robustas para Moran/LISA.

Salida: grilla_h3_res8.gpkg (poligonos + variables agregadas).

Uso:  python grilla_h3.py
"""
import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

RES = 8
ENTRADA_REPORTES = "dataset_analitico.parquet"
ENTRADA_POIS = "dataset_pois.csv"
ENTRADA_COMUNAS = "comunas.geojson"
SALIDA = "grilla_h3_res8.gpkg"
CRS_GEO = "EPSG:4326"


def celda_de(lat, lon, res=RES):
    return h3.latlng_to_cell(lat, lon, res)


# --------------------------------------------------------------------------- #
# 1. Teselado de la zona de estudio (cobertura completa, incluye celdas con 0)
# --------------------------------------------------------------------------- #
def teselar_comunas(res=RES) -> pd.DataFrame:
    com = gpd.read_file(ENTRADA_COMUNAS).to_crs(CRS_GEO)
    filas = []
    for _, r in com.iterrows():
        geom = r.geometry
        partes = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in partes:
            ext = [(y, x) for x, y in p.exterior.coords]  # H3 usa (lat, lng)
            huecos = [[(y, x) for x, y in ring.coords] for ring in p.interiors]
            poly = h3.LatLngPoly(ext, *huecos) if huecos else h3.LatLngPoly(ext)
            for c in h3.polygon_to_cells(poly, res):
                filas.append((c, r["comuna"]))
    tes = pd.DataFrame(filas, columns=["h3", "comuna"]).drop_duplicates("h3")
    print(f"[1] Teselado: {len(tes):,} celdas H3 res{res} cubren las 5 comunas")
    return tes


# --------------------------------------------------------------------------- #
# 2. Agregacion de reportes por celda
# --------------------------------------------------------------------------- #
def agregar_reportes(res=RES, filtro_fecha=None) -> pd.DataFrame:
    """filtro_fecha: tupla (inicio, fin) 'YYYY-MM-DD' para acotar el periodo
    (p.ej. ('2024-01-01', '2024-01-31') = solo enero 2024). None = todo el periodo."""
    df = pd.read_parquet(ENTRADA_REPORTES, engine="fastparquet",
                         columns=["lat", "lon", "es_delito", "nocturno",
                                  "fin_de_semana", "comuna", "fecha"])
    if filtro_fecha is not None:
        ini, fin = filtro_fecha
        df = df[(df["fecha"] >= ini) & (df["fecha"] <= fin)]
        print(f"    filtro de fechas {ini} a {fin}: {len(df):,} reportes")
    df["h3"] = [celda_de(la, lo, res) for la, lo in zip(df["lat"], df["lon"])]
    g = df.groupby("h3")
    agg = pd.DataFrame({
        "n_reportes": g.size(),
        "n_delito": g["es_delito"].sum(),
        "n_nocturno": g["nocturno"].sum(),
        "n_delito_nocturno": g.apply(lambda x: (x["es_delito"] & x["nocturno"]).sum(),
                                     include_groups=False),
        "comuna_rep": g["comuna"].agg(lambda s: s.mode().iat[0]),
    })
    print(f"[2] Reportes agregados: {len(agg):,} celdas con al menos 1 reporte")
    return agg


# --------------------------------------------------------------------------- #
# 3. Agregacion de POIs por celda (conteo por categoria)
# --------------------------------------------------------------------------- #
def agregar_pois(res=RES) -> pd.DataFrame:
    p = pd.read_csv(ENTRADA_POIS)
    p["h3"] = [celda_de(la, lo, res) for la, lo in zip(p["Latitud"], p["Longitud"])]
    piv = (p.groupby(["h3", "Categoria"]).size().unstack(fill_value=0)
           .rename(columns=lambda c: "n_" + c.replace(" ", "_")))
    piv["n_pois_total"] = piv.sum(axis=1)
    print(f"[3] POIs agregados: {len(piv):,} celdas con al menos 1 POI  "
          f"({list(piv.columns)})")
    return piv


# --------------------------------------------------------------------------- #
# 4. Construir GeoDataFrame de la grilla
# --------------------------------------------------------------------------- #
def construir_grilla(res=RES, filtro_fecha=None) -> gpd.GeoDataFrame:
    tes = teselar_comunas(res).set_index("h3")
    rep = agregar_reportes(res, filtro_fecha=filtro_fecha)
    poi = agregar_pois(res)

    # base = teselado U celdas con reportes U celdas con POIs (no perder datos de borde)
    celdas = tes.index.union(rep.index).union(poi.index)
    g = pd.DataFrame(index=celdas)
    g = g.join(tes).join(rep).join(poi)

    # comuna: la del teselado; si la celda es de borde (solo reportes), la modal de reportes
    g["comuna"] = g["comuna"].fillna(g["comuna_rep"])
    g = g.drop(columns="comuna_rep")

    # rellenar conteos faltantes con 0 (celdas sin reportes o sin POIs)
    cols_cont = ["n_reportes", "n_delito", "n_nocturno", "n_delito_nocturno",
                 "n_botilleria", "n_comisaria", "n_discoteca",
                 "n_estacion_de_metro", "n_pois_total"]
    for c in cols_cont:
        if c in g:
            g[c] = g[c].fillna(0).astype(int)
        else:
            g[c] = 0

    # variables derivadas a nivel celda
    g["tasa_delito"] = np.where(g["n_reportes"] > 0, g["n_delito"] / g["n_reportes"], np.nan)
    g["area_km2"] = [h3.cell_area(c, unit="km^2") for c in g.index]
    g["dens_delito_km2"] = g["n_delito"] / g["area_km2"]
    g["dens_reportes_km2"] = g["n_reportes"] / g["area_km2"]
    g["dens_botilleria_km2"] = g["n_botilleria"] / g["area_km2"]
    g["prop_nocturno"] = np.where(g["n_reportes"] > 0, g["n_nocturno"] / g["n_reportes"], np.nan)

    # geometria del hexagono
    geom = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in g.index]
    gdf = gpd.GeoDataFrame(g.reset_index(names="h3"), geometry=geom, crs=CRS_GEO)

    print(f"[4] Grilla: {len(gdf):,} celdas | con reportes={int((gdf.n_reportes>0).sum()):,} "
          f"| con POIs={int((gdf.n_pois_total>0).sum()):,}")
    return gdf


def main():
    gdf = construir_grilla()
    gdf.to_file(SALIDA, driver="GPKG")
    print(f"[5] Guardado: {SALIDA}  ({len(gdf):,} celdas, {gdf.shape[1]} columnas)")
    print("\nResumen por comuna:")
    print(gdf.groupby("comuna").agg(
        celdas=("h3", "size"),
        reportes=("n_reportes", "sum"),
        delitos=("n_delito", "sum"),
        tasa_delito_media=("tasa_delito", "mean"),
        botillerias=("n_botilleria", "sum"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
