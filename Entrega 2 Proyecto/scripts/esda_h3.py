"""
Punto 3 - Analisis exploratorio espacial: autocorrelacion global (I de Moran)
y local (LISA) sobre la grilla H3 res 8.

Matriz de pesos W: vecindad H3 (los 6 hexagonos adyacentes, h3.grid_disk k=1),
normalizada por fila ('r'). El analisis se restringe a celdas urbanizadas
(n_reportes >= MIN_REP) para evitar el artefacto de las celdas vacias (cerros,
parques) y la inestabilidad de tasas con pocos reportes.

Uso:  python esda_h3.py
"""
import h3
import numpy as np
import geopandas as gpd
from libpysal.weights import W
from esda.moran import Moran, Moran_Local

GRILLA = "grilla_h3_res8.gpkg"
MIN_REP = 30          # celdas con al menos 30 reportes (tasas estables, area urbana)
VAR = "dens_delito_km2"
PERM = 999
SEED = 42


def cargar(min_rep=MIN_REP) -> gpd.GeoDataFrame:
    g = gpd.read_file(GRILLA)
    g = g[g["n_reportes"] >= min_rep].reset_index(drop=True)
    return g


def construir_w(g) -> W:
    """W de contiguidad H3 (k=1) restringida a las celdas presentes; sin islas."""
    celdas = set(g["h3"])
    neighbors = {c: [n for n in h3.grid_disk(c, 1) if n != c and n in celdas]
                 for c in g["h3"]}
    # eliminar islas (celdas sin vecinos) para un grafo conexo
    islas = [c for c, nb in neighbors.items() if len(nb) == 0]
    if islas:
        for c in islas:
            neighbors.pop(c)
        neighbors = {c: [n for n in nb if n not in islas] for c, nb in neighbors.items()}
    w = W(neighbors, silence_warnings=True)
    w.transform = "r"
    return w, islas


def analizar_gdf(g, var=VAR, min_rep=MIN_REP):
    """Moran global + LISA sobre una grilla H3 cualquiera (res 8, 9, ...)."""
    g = g[g["n_reportes"] >= min_rep].reset_index(drop=True)
    w, islas = construir_w(g)
    g = g[g["h3"].isin(w.id_order)].set_index("h3").loc[w.id_order].reset_index()
    y = g[var].values.astype(float)
    np.random.seed(SEED)  # reproducibilidad de las permutaciones

    # --- Moran global ---
    mi = Moran(y, w, permutations=PERM)
    print(f"[Moran global] variable='{var}'  n={len(g)} celdas  islas removidas={len(islas)}")
    print(f"   I = {mi.I:.3f}   E[I] = {mi.EI:.3f}   z = {mi.z_sim:.2f}   "
          f"p(perm) = {mi.p_sim:.4f}")

    # --- LISA local ---
    lisa = Moran_Local(y, w, permutations=PERM)
    sig = lisa.p_sim < 0.05
    etiquetas = {1: "Alto-Alto (hotspot)", 2: "Bajo-Alto", 3: "Bajo-Bajo (coldspot)", 4: "Alto-Bajo"}
    g["lisa_q"] = lisa.q
    g["lisa_sig"] = sig
    g["lisa_cluster"] = np.where(sig, [etiquetas[q] for q in lisa.q], "No significativo")

    print(f"\n[LISA] clusters significativos (p<0.05): {int(sig.sum())} de {len(g)} celdas")
    print(g["lisa_cluster"].value_counts().to_string())
    return g, w, mi, lisa


def analizar(var=VAR, min_rep=MIN_REP):
    """Atajo: lee la grilla res 8 cacheada y analiza."""
    return analizar_gdf(cargar(min_rep), var, min_rep)


if __name__ == "__main__":
    analizar()
