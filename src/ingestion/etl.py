"""
etl.py
------
Pipeline unificado MEPS 2018-2023 (longitudinal).
Lee H251 + H249 para cada año, cachea en CSV para ejecuciones rápidas,
produce un DataFrame consolidado con columna 'año'.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "raw" / "cache"
DATA_PROCESSED = ROOT / "data" / "processed"

CODIGOS_RESERVADOS = [-1, -7, -8, -9, -15]

# =========================================================
# CONFIGURACIÓN POR AÑO
# =========================================================
ARCHIVOS_MEPS = {
    2018: {"h251": "h209.xlsx", "h249": "h207.xlsx", "sufijo": "18"},
    2019: {"h251": "h216.xlsx", "h249": "h214.xlsx", "sufijo": "19"},
    2020: {"h251": "h224.xlsx", "h249": "h222.xlsx", "sufijo": "20"},
    2021: {"h251": "h233.xlsx", "h249": "h231.xlsx", "sufijo": "21"},
    2022: {"h251": "h243.xlsx", "h249": "h241.xlsx", "sufijo": "22"},
    2023: {"h251": "h251.xlsx", "h249": "h249.xlsx", "sufijo": "23"},
}

# =========================================================
# COLUMNAS FIJAS (sin sufijo — iguales en todos los años)
# =========================================================
COLS_FIJAS_H251 = {
    "DUPERSID":   "persona_id",
    "PANEL":      "panel",
    "VARSTR":     "varianza_estrato",
    "VARPSU":     "varianza_psu",
    "SEX":        "sexo",
    "RACEV2X":    "raza",
    "BORNUSA":    "nacido_usa",
    "HIBPDX":     "dx_hipertension",
    "DIABDX_M18": "dx_diabetes",
    "ASTHDX":     "dx_asma",
    "CANCERDX":   "dx_cancer",
    "ARTHDX":     "dx_artritis",
    "CHDDX":      "dx_cardiopatia",
    "STRKDX":     "dx_ictus",
    "EMPHDX":     "dx_enfisema",
}

# =========================================================
# COLUMNAS CON SUFIJO (cambian por año)
# =========================================================
COLS_SUFIJO_H251 = {
    "PERWT{s}F":  "peso_muestral",
    "AGE{s}X":    "edad",
    "MARRY{s}X":  "estado_civil",
    "REGION{s}":  "region",
    "POVCAT{s}":  "categoria_pobreza",
    "POVLEV{s}":  "pct_nivel_pobreza",
    "TTLP{s}X":   "ingreso_personal",
    "FAMINC{s}":  "ingreso_familiar",
    "INSCOV{s}":  "tipo_cobertura",
    "UNINS{s}":   "sin_seguro_anual",
    "RTHLTH53":   "salud_general",
    "MNHLTH53":   "salud_mental",
    "TOTEXP{s}":  "gasto_total_anual",
    "TOTSLF{s}":  "gasto_bolsillo",
    "TOTPTR{s}":  "gasto_seguro_privado",
    "TOTMCR{s}":  "gasto_medicare",
    "TOTMCD{s}":  "gasto_medicaid",
    "RXEXP{s}":   "gasto_medicamentos",
    "OBTOTV{s}":  "visitas_ambulatorias",
    "OPTOTV{s}":  "visitas_outpatient",
    "ERTOT{s}":   "visitas_urgencias",
    "ERDEXP{s}":  "gasto_urgencias",
    "IPNGTD{s}":  "noches_hospital",
    "RXTOT{s}":   "total_recetas",
    "HIDEG":      "max_nivel_educativo",
    "EDUCYR":     "años_educacion",
}

# =========================================================
# COLUMNAS H249
# =========================================================
COLS_H249 = {
    "DUPERSID":  "persona_id",
    "ICD10CDX":  "diagnostico_icd10",
    "CCSR1X":    "categoria_clinica_1",
    "CCSR2X":    "categoria_clinica_2",
    "CCSR3X":    "categoria_clinica_3",
    "CCSR4X":    "categoria_clinica_4",
    "INJURY":    "condicion_por_lesion",
    "AGEDIAG":   "edad_diagnostico_condicion",
    "ERCOND":    "condicion_relacionada_urgencias",
    "IPCOND":    "condicion_relacionada_hospitalizacion",
    "OBCOND":    "condicion_relacionada_consulta",
    "OPCOND":    "condicion_relacionada_outpatient",
    "RXCOND":    "condicion_relacionada_medicacion",
    "HHCOND":    "condicion_relacionada_homecare",
    "ACCDNWRK":  "accidente_ocurrido_trabajo",
    "ERNUM": "num_eventos_urgencias",
    "IPNUM": "num_eventos_hospitalizacion",
    "RXNUM": "num_eventos_medicacion",
}

FLAGS_DX = [
    "dx_hipertension", "dx_diabetes", "dx_asma", "dx_cancer",
    "dx_artritis", "dx_cardiopatia", "dx_ictus", "dx_enfisema",
]

COLUMNAS_TEXTO_H249 = [
    "diagnostico_icd10", "categoria_clinica_1", "categoria_clinica_2",
    "categoria_clinica_3", "categoria_clinica_4",
]


# =========================================================
# UTIL
# =========================================================
def clean(df):
    return df.replace(CODIGOS_RESERVADOS, np.nan)


def leer_con_cache(path_xlsx: Path, path_csv: Path) -> pd.DataFrame:
    """Lee desde cache CSV si existe, sino lee xlsx y cachea."""
    if path_csv.exists():
        print(f"  [cache] Leyendo {path_csv.name}")
        return pd.read_csv(path_csv, low_memory=False)
    print(f"  [xlsx] Leyendo {path_xlsx.name} (primera vez, puede tardar...)")
    df = pd.read_excel(path_xlsx)
    df.to_csv(path_csv, index=False)
    print(f"  [cache] Guardado {path_csv.name}")
    return df


def build_col_map_h251(sufijo: str) -> dict:
    """Construye el mapeo completo de columnas H251 para un año dado."""
    col_map = dict(COLS_FIJAS_H251)
    for patron, nombre in COLS_SUFIJO_H251.items():
        col_real = patron.replace("{s}", sufijo)
        col_map[col_real] = nombre
    return col_map


# =========================================================
# LOAD H251 — UN AÑO
# =========================================================
def load_h251_año(año: int) -> pd.DataFrame:
    cfg = ARCHIVOS_MEPS[año]
    sufijo = cfg["sufijo"]

    path_xlsx = DATA_RAW / cfg["h251"]
    path_csv = DATA_CACHE / f"h251_{año}.csv"

    df = leer_con_cache(path_xlsx, path_csv)

    col_map = build_col_map_h251(sufijo)
    faltantes = [k for k in col_map if k not in df.columns]
    if faltantes:
        print(f"  [ADVERTENCIA] H251 {año} — no encontradas: {faltantes}")

    cols_disponibles = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(cols_disponibles.keys())].rename(columns=cols_disponibles)

    for c in df.columns:
        if c != "persona_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = clean(df)

    # Recodificar flags diagnóstico: 1=Sí → 1, 2=No → 0
    for col in FLAGS_DX:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 2: 0})

    df["año"] = año
    return df


# =========================================================
# LOAD H249 — UN AÑO
# =========================================================
def load_h249_año(año: int) -> pd.DataFrame:
    cfg = ARCHIVOS_MEPS[año]

    path_xlsx = DATA_RAW / cfg["h249"]
    path_csv = DATA_CACHE / f"h249_{año}.csv"

    df = leer_con_cache(path_xlsx, path_csv)

    faltantes = [k for k in COLS_H249 if k not in df.columns]
    if faltantes:
        print(f"  [ADVERTENCIA] H249 {año} — no encontradas: {faltantes}")

    cols_disponibles = {k: v for k, v in COLS_H249.items() if k in df.columns}
    df = df[list(cols_disponibles.keys())].rename(columns=cols_disponibles)

    for c in df.columns:
        if c != "persona_id" and c not in COLUMNAS_TEXTO_H249:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return clean(df)



# =========================================================
# PIPELINE UN AÑO
# =========================================================
def aggregate_h249_año(df: pd.DataFrame) -> pd.DataFrame:

    agg_dict = {
        "n_diagnosticos_unicos": ("diagnostico_icd10", "nunique"),
        "n_sistemas_afectados":  ("categoria_clinica_1", "nunique"),
        "lesiones":              ("condicion_por_lesion", "sum"),
        "accidentes_trabajo":    ("accidente_ocurrido_trabajo", "sum"),
    }

    # er_eventos — ERCOND (2021+) o ERNUM (2018-2020)
    if "condicion_relacionada_urgencias" in df.columns:
        agg_dict["er_eventos"] = ("condicion_relacionada_urgencias", "sum")
    elif "num_eventos_urgencias" in df.columns:
        agg_dict["er_eventos"] = ("num_eventos_urgencias", "sum")

    # hosp_eventos — IPCOND (2021+) o IPNUM (2018-2020)
    if "condicion_relacionada_hospitalizacion" in df.columns:
        agg_dict["hosp_eventos"] = ("condicion_relacionada_hospitalizacion", "sum")
    elif "num_eventos_hospitalizacion" in df.columns:
        agg_dict["hosp_eventos"] = ("num_eventos_hospitalizacion", "sum")

    # rx_eventos — RXCOND (2021+) o RXNUM (2018-2020)
    if "condicion_relacionada_medicacion" in df.columns:
        agg_dict["rx_eventos"] = ("condicion_relacionada_medicacion", "sum")
    elif "num_eventos_medicacion" in df.columns:
        agg_dict["rx_eventos"] = ("num_eventos_medicacion", "sum")

    return df.groupby("persona_id").agg(**agg_dict).reset_index()




def load_meps_año(año: int) -> pd.DataFrame:
    print(f"\n[etl] Procesando {año}...")
    h251 = load_h251_año(año)
    h249 = load_h249_año(año)
    h249_agg = aggregate_h249_año(h249)

    df = h251.merge(h249_agg, on="persona_id", how="left")

    conteos = ["n_diagnosticos_unicos", "n_sistemas_afectados",
               "er_eventos", "hosp_eventos", "rx_eventos",
               "lesiones", "accidentes_trabajo"]
    for col in conteos:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["sexo_femenino"] = (df["sexo"] == 2).astype("Int8")
    df["comorbilidad"] = df["n_diagnosticos_unicos"].fillna(0)
    df["alto_coste"] = (
        df["gasto_total_anual"] >= df["gasto_total_anual"].quantile(0.80)
    ).astype("Int8")
    df["salud_pobre"] = (df["salud_general"] >= 4).astype("Int8")
    df["log_gasto"] = np.log1p(df["gasto_total_anual"])

    print(f"  [etl] {año} — {len(df):,} personas, {df.shape[1]} columnas")
    return df


# =========================================================
# PIPELINE COMPLETO — TODOS LOS AÑOS
# =========================================================
def load_meps(anios: list = None) -> pd.DataFrame:
    if anios is None:
        anios = list(ARCHIVOS_MEPS.keys())

    dfs = []
    for año in anios:
        df_año = load_meps_año(año)
        dfs.append(df_año)

    df_total = pd.concat(dfs, ignore_index=True)

    path_salida = DATA_PROCESSED / "meps_2018_2023.csv"
    df_total.to_csv(path_salida, index=False)
    print(f"\n[etl] Guardado {path_salida.name} — {len(df_total):,} filas, {df_total.shape[1]} columnas")
    print(f"[etl] Años: {df_total['año'].value_counts().sort_index().to_dict()}")

    
    df_2023 = df_total[df_total["año"] == 2023].reset_index(drop=True)
    path_2023 = DATA_PROCESSED / "meps_2023_clean.csv"
    df_2023.to_csv(path_2023, index=False)
    print(f"[etl] Guardado {path_2023.name} — {len(df_2023):,} filas")

    return df_total


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = load_meps()
    print(df.head())
    print(df.dtypes)
    print(df["gasto_total_anual"].describe())