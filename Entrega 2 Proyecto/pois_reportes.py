"""
Para cada POI: cuantos reportes (y delitos) SOSAFE hay a su alrededor (radios en metros).
Es el INVERSO de las variables del pipeline (que cuentan POIs cerca de cada reporte).

Salida: dataset_pois_enriquecido.csv  (un POI por fila + conteos de reportes/delitos).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

CRS_METRICO = "EPSG:32719"
RADIOS = (100, 200, 500)


def enriquecer_pois():
    # --- POIs (puntos de busqueda) ---
    p = pd.read_csv("dataset_pois.csv")
    gp = gpd.GeoDataFrame(p, geometry=gpd.points_from_xy(p.Longitud, p.Latitud),
                          crs="EPSG:4326").to_crs(CRS_METRICO)
    xy_poi = np.c_[gp.geometry.x, gp.geometry.y]

    # --- Reportes (ya limpios, en las 5 comunas) ---
    r = pd.read_parquet("dataset_analitico.parquet", engine="fastparquet",
                        columns=["lon", "lat", "es_delito"])
    gr = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r.lon, r.lat),
                          crs="EPSG:4326").to_crs(CRS_METRICO)
    xy_rep = np.c_[gr.geometry.x, gr.geometry.y]
    xy_del = xy_rep[r["es_delito"].values]

    tree_rep = cKDTree(xy_rep)
    tree_del = cKDTree(xy_del)

    for radio in RADIOS:
        p[f"n_reportes_{radio}m"] = tree_rep.query_ball_point(xy_poi, radio, return_length=True)
        p[f"n_delitos_{radio}m"] = tree_del.query_ball_point(xy_poi, radio, return_length=True)
    p["tasa_delito_200m"] = (p["n_delitos_200m"] / p["n_reportes_200m"]).round(3)

    p.to_csv("dataset_pois_enriquecido.csv", index=False, encoding="utf-8-sig")
    return p


if __name__ == "__main__":
    p = enriquecer_pois()
    print(f"POIs: {len(p)}  -> dataset_pois_enriquecido.csv\n")

    print("=== Promedio de reportes/delitos en 200 m, por categoria ===")
    print(p.groupby("Categoria")[["n_reportes_200m", "n_delitos_200m", "tasa_delito_200m"]]
          .mean().round(1).to_string())

    print("\n=== Top 10 POIs con MAS delitos en 200 m ===")
    cols = ["Nombre", "Categoria", "Comuna_Origen_Busqueda", "n_reportes_200m", "n_delitos_200m"]
    print(p.sort_values("n_delitos_200m", ascending=False)[cols].head(10).to_string(index=False))