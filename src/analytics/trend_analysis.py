"""
trend_analysis.py
------------------
Análisis de tendencias de gasto y utilización médica, MEPS 2018-2023.
Usa el dataset longitudinal consolidado (meps_2018_2023.csv).

Metodología:
1. Tendencia cruda (ponderada) de gasto y utilización por año
2. Aislamiento del efecto COVID (2020) como nota, no como outlier a eliminar
3. GLM con 'año' como covariable, controlando por mix demográfico,
   para separar tendencia real de cambio en composición de la muestra
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Columnas estables en los 6 años (evita CCSR4X y los CONDs de disponibilidad parcial)
COLS_UTILIZACION = [
    "visitas_ambulatorias", "visitas_outpatient",
    "visitas_urgencias", "noches_hospital", "total_recetas",
]

# Features demográficas para controlar mix de muestra en el GLM de tendencia
FEATURES_CONTROL = [
    "edad", "sexo_femenino", "raza", "estado_civil", "region",
    "años_educacion", "categoria_pobreza", "ingreso_familiar",
    "tipo_cobertura", "sin_seguro_anual",
]


# =========================================================
# 1. TENDENCIA CRUDA PONDERADA — GASTO Y % CEROS
# =========================================================
def tendencia_gasto_anual(df, col="gasto_total_anual", peso="peso_muestral"):
    """
    Media y mediana ponderadas de gasto por año, más % de gasto cero.
    La ponderación es transversal por año (los pesos MEPS no se suman
    entre años sin normalizar primero).
    """
    resultados = []
    for año, g in df.groupby("año"):
        w = g[peso].values
        w = w / w.mean()
        y = g[col].values

        media_pond = np.average(y, weights=w)
        # mediana ponderada
        orden = np.argsort(y)
        y_ord, w_ord = y[orden], w[orden]
        cum_w = np.cumsum(w_ord) / np.sum(w_ord)
        mediana_pond = y_ord[np.searchsorted(cum_w, 0.5)]

        pct_cero = np.average((y == 0).astype(float), weights=w) * 100

        resultados.append({
            "año": año,
            "n": len(g),
            "gasto_medio": round(media_pond, 1),
            "gasto_mediana": round(mediana_pond, 1),
            "pct_gasto_cero": round(pct_cero, 2),
        })

    tabla = pd.DataFrame(resultados).sort_values("año").reset_index(drop=True)
    print("\nTendencia de gasto anual (ponderada):")
    print(tabla.to_string(index=False))
    print("\n→ 2020 puede mostrar caída por efecto COVID (reducción de utilización electiva)")
    print("→ No se elimina como outlier: es una observación real del periodo")

    return tabla


# =========================================================
# 2. TENDENCIA DE UTILIZACIÓN
# =========================================================
def tendencia_utilizacion(df, peso="peso_muestral"):
    """
    Evolución anual ponderada de visitas, urgencias, hospitalización y recetas.
    """
    resultados = []
    for año, g in df.groupby("año"):
        w = g[peso].values
        w = w / w.mean()

        fila = {"año": año}
        for col in COLS_UTILIZACION:
            if col in g.columns:
                fila[col] = round(np.average(g[col].fillna(0), weights=w), 2)
        resultados.append(fila)

    tabla = pd.DataFrame(resultados).sort_values("año").reset_index(drop=True)
    print("\nTendencia de utilización anual (ponderada):")
    print(tabla.to_string(index=False))

    return tabla


# =========================================================
# 3. GLM DE TENDENCIA — CONTROLANDO MIX DEMOGRÁFICO
# =========================================================
def glm_tendencia_controlada(df, col="gasto_total_anual", peso="peso_muestral"):
    """
    GLM Gamma con 'año' como covariable (dummies), controlando por
    features demográficas, para aislar la tendencia real de gasto del
    cambio en composición de la muestra año a año.

    Usa la misma familia Gamma seleccionada formalmente en
    diagnostico_distribucion.py (ver decisiones_tecnicas.md).
    """
    cols_necesarias = FEATURES_CONTROL + [col, peso, "año"]
    df_model = df[cols_necesarias].dropna()
    df_model = df_model[df_model[col] > 0].reset_index(drop=True)

    df_model["año"] = df_model["año"].astype(int)
    dummies_año = pd.get_dummies(df_model["año"], prefix="año", drop_first=True, dtype=float)

    X = pd.concat([df_model[FEATURES_CONTROL], dummies_año], axis=1)
    X = sm.add_constant(X)
    y = df_model[col].astype(float).values
    w = df_model[peso].values
    w = w / w.mean()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = sm.GLM(
            y, X,
            family=sm.families.Gamma(link=sm.families.links.Log()),
            freq_weights=w
        ).fit()

    cols_año = [c for c in X.columns if c.startswith("año_")]
    resumen = pd.DataFrame({
        "año_vs_2018": [c.replace("año_", "") for c in cols_año],
        "coeficiente": modelo.params[cols_año].round(4).values,
        "exp_coef": np.exp(modelo.params[cols_año]).round(4).values,
        "p_valor": modelo.pvalues[cols_año].round(4).values,
    })
    resumen["significativo"] = resumen["p_valor"] < 0.05
    resumen["interpretacion"] = resumen["exp_coef"].apply(
        lambda x: f"{(x - 1) * 100:+.1f}% vs 2018"
    )

    print("\nGLM Gamma — tendencia de gasto controlando mix demográfico (base: 2018):")
    print(resumen.to_string(index=False))
    print("\n→ Esto es la tendencia 'real' — aísla el efecto año del cambio en composición de la muestra")
    print("→ Comparar con tendencia_gasto_anual() para ver cuánto explica el mix demográfico")

    return modelo, resumen


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2018_2023.csv")

    print(">>> Tendencia cruda de gasto...")
    tabla_gasto = tendencia_gasto_anual(df)

    print("\n>>> Tendencia de utilización...")
    tabla_util = tendencia_utilizacion(df)

    print("\n>>> GLM de tendencia controlada por mix demográfico...")
    modelo_tend, resumen_tend = glm_tendencia_controlada(df)