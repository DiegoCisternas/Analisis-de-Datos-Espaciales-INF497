"""
Punto 3 (extension) - Analisis ESPACIO-TEMPORAL: como cambian la I de Moran
global y los clusters LISA segun la rebanada temporal (hora, dia, estacion) y
en fechas especiales (18-Sep, Ano Nuevo).

Diseno: se fija UNA sola unidad espacial y matriz de pesos W (las 396 celdas
urbanas con >=30 reportes del Punto 3, vecindad H3 k=1). Para cada rebanada solo
cambian los CONTEOS de delito por celda -> las comparaciones de I/LISA son
directas (mismo soporte espacial, misma W).

Uso:  python esda_temporal.py
"""
import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from esda.moran import Moran, Moran_Local
import esda_h3 as E

RES = 8
PERM = 999
SEED = 42
REPORTES = "dataset_analitico.parquet"


def preparar():
    """Puntos con celda H3 res8, y la grilla fija + W del Punto 3."""
    g = E.cargar()                       # 396 celdas con >=30 reportes
    w, islas = E.construir_w(g)
    g = g[g["h3"].isin(w.id_order)].set_index("h3").loc[w.id_order]
    area = g["area_km2"].values
    geom = gpd.GeoDataFrame(g.reset_index(), geometry=g.geometry.values, crs=g.crs)

    pts = pd.read_parquet(REPORTES, engine="fastparquet",
                          columns=["lat", "lon", "es_delito", "fecha", "mes", "anio",
                                   "nocturno", "fin_de_semana", "periodo"])
    pts["fecha"] = pd.to_datetime(pts["fecha"])
    pts["h3"] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(pts["lat"], pts["lon"])]
    pts = pts[pts["h3"].isin(w.id_order)]          # solo celdas del soporte fijo
    return pts, w, area, geom


def densidad_slice(pts, mask, w, area):
    """Vector de densidad de delito por celda (alineado a w.id_order) para la rebanada."""
    sub = pts[mask & pts["es_delito"]]
    cnt = sub.groupby("h3").size().reindex(w.id_order, fill_value=0)
    return cnt.values / area, int(cnt.sum())


def morlisa(pts, mask, w, area, label, lisa=True):
    np.random.seed(SEED)
    y, n = densidad_slice(pts, mask, w, area)
    mi = Moran(y, w, permutations=PERM)
    res = {"slice": label, "n_delito": n, "I": round(mi.I, 3),
           "p": round(mi.p_sim, 4), "z": round(mi.z_sim, 2)}
    out = {"y": y, "moran": mi}
    if lisa:
        ml = Moran_Local(y, w, permutations=PERM)
        sig = ml.p_sim < 0.05
        res["n_hotspot"] = int(((ml.q == 1) & sig).sum())
        res["n_coldspot"] = int(((ml.q == 3) & sig).sum())
        out["lisa"] = ml
    return res, out


def definir_slices(pts):
    return {
        "Nocturno (20-05h)":     pts["nocturno"],
        "Diurno (06-19h)":       ~pts["nocturno"],
        "Fin de semana":         pts["fin_de_semana"],
        "Dia de semana":         ~pts["fin_de_semana"],
        "Verano 2024 (D-E-F)":   (pts["anio"] == 2024) & pts["mes"].isin([12, 1, 2]),
        "Invierno 2024 (J-J-A)": (pts["anio"] == 2024) & pts["mes"].isin([6, 7, 8]),
        "Madrugada (0-5h)":      pts["periodo"] == "madrugada",
        "Manana (6-11h)":        pts["periodo"] == "manana",
        "Tarde (12-18h)":        pts["periodo"] == "tarde",
        "Noche (19-23h)":        pts["periodo"] == "noche",
    }


def slices_especiales(pts):
    f = pts["fecha"]
    return {
        "18-Sep 2024 (17-20)": (f >= "2024-09-17") & (f <= "2024-09-20"),
        "Ano Nuevo (31/12-01/01)": (((f >= "2023-12-31") & (f <= "2024-01-01")) |
                                    ((f >= "2024-12-31") & (f <= "2025-01-01"))),
        "Dia normal (referencia)": pts["fecha"].dt.dayofweek.isin([1, 2, 3]) &
                                   ~pts["mes"].isin([12, 1, 2, 9]),
    }


def mapa_lisa(out, geom, com, ax, title):
    from splot.esda import lisa_cluster
    lisa_cluster(out["lisa"], geom, p=0.05, ax=ax, legend=True)
    com.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_title(title); ax.set_axis_off()


def mapa_densidad(y, n, geom, com, ax, title):
    gg = geom.copy(); gg["dens"] = y
    gg.plot(column="dens", scheme="quantiles", k=5, cmap="OrRd", legend=True,
            legend_kwds={"fmt": "{:.0f}", "fontsize": 6}, edgecolor="grey",
            linewidth=0.1, ax=ax)
    com.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_title(f"{title} (n={n})"); ax.set_axis_off()


def main():
    pts, w, area, geom = preparar()
    print(f"puntos en soporte fijo: {len(pts):,}  | celdas: {w.n}\n")
    filas = []
    for label, mask in definir_slices(pts).items():
        res, _ = morlisa(pts, mask, w, area, label)
        filas.append(res)
    for label, mask in slices_especiales(pts).items():
        res, _ = morlisa(pts, mask, w, area, label, lisa=False)  # solo Moran global
        filas.append(res)
    tab = pd.DataFrame(filas)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
