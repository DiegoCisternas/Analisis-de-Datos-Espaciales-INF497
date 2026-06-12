"""
EDA temporal/categorico (no espacial) de los DELITOS.

Define un subtipo de delito mas fino que `categoria_inferida` combinando el
codigo `type` (mas confiable) con grupos de keywords, y carga el subconjunto
es_delito listo para analisis temporal.

Subtipos: asalto_violento, robo_hurto, armas_disparos, drogas, vandalismo,
violencia_intrafamiliar, sospechoso, otro_delito.

Nota: la violencia intrafamiliar queda casi sin detectar (los reportes no usan
keywords de VIF) -> subdeteccion conocida (limitacion).
"""
import pandas as pd
from pipeline_sosafe import quita_tildes

# orden de prioridad: el primer match asigna el subtipo
GRUPOS_KW = [
    ("armas_disparos", r"\b(?:balacera|balazo|balea|disparo|tiroteo|arma de fuego|"
                       r"arma blanca|pistola|revolver|escopeta|navaja|punal|municion)\w*"),
    ("asalto_violento", r"\b(?:asalt|atraco|cogote|apunal|acuchill|asesin|homicid|"
                        r"secuestr|raptaron|encerrona|portonazo|amenaz)\w*"),
    ("robo_hurto",      r"\b(?:robo|roban|robaron|robando|robar|hurto|ladron|arrebat|"
                        r"lanzazo|turbazo|motochorro|mechero|carterist|monre|alunizaje)\w*"),
    ("drogas",          r"\b(?:droga|narco|microtrafico|traficante|pasta base)\w*"),
    ("vandalismo",      r"\b(?:vandaliz|vandalico|saqueo|saquearon|saquean)\w*"),
]
MAPA_TYPE_DELITO = {
    0: "asalto_violento", 146: "asalto_violento",
    1: "robo_hurto", 2: "robo_hurto", 112: "robo_hurto", 139: "robo_hurto",
    28: "vandalismo", 118: "violencia_intrafamiliar", 48: "violencia_intrafamiliar",
}
ORDEN_SUBTIPOS = ["robo_hurto", "asalto_violento", "armas_disparos", "drogas",
                  "sospechoso", "vandalismo", "violencia_intrafamiliar", "otro_delito"]


def clasificar_subtipo(d: pd.DataFrame) -> pd.Series:
    txt = quita_tildes(d["description"])
    tipo = pd.Series("otro_delito", index=d.index)
    ya = pd.Series(False, index=d.index)
    for code, t in MAPA_TYPE_DELITO.items():       # 1) por type (mas confiable)
        m = (d["type"] == code) & ~ya
        tipo[m] = t; ya |= m
    for nombre, pat in GRUPOS_KW:                  # 2) por keyword
        m = ~ya & txt.str.contains(pat, regex=True, na=False)
        tipo[m] = nombre; ya |= m
    m = ~ya & (d["type"] == 6)                     # 3) sospechoso (type 6 restante)
    tipo[m] = "sospechoso"
    return tipo


def cargar_delitos(parquet="dataset_analitico.parquet") -> pd.DataFrame:
    cols = ["type", "description", "categoria_inferida", "es_delito", "hour", "dow",
            "mes", "anio", "fecha", "fin_de_semana", "nocturno", "periodo",
            "comuna", "likes", "comments"]
    df = pd.read_parquet(parquet, engine="fastparquet", columns=cols)
    d = df[df["es_delito"]].copy()
    d["tipo_delito"] = clasificar_subtipo(d)
    d["fecha"] = pd.to_datetime(d["fecha"])
    dias = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    d["dia_sem"] = d["dow"].map(dict(enumerate(dias)))
    return d


if __name__ == "__main__":
    d = cargar_delitos()
    print(f"delitos: {len(d):,}")
    print(d["tipo_delito"].value_counts().to_string())
