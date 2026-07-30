"""
explainability.py
--------------------
SHAP global + local para los modelos ya entrenados en el proyecto.
Primer módulo de la Parte 4 (Explainability & business insights).

Cobertura:
1. Modelo LightGBM de alto riesgo (risk_detection.py) — TreeExplainer,
   exacto y rápido para modelos de árboles.
2. Modelo Gamma de severidad (claim_sev.py) — KernelExplainer (model-
   agnostic), necesario porque SHAP no tiene explainer nativo para GLM
   de statsmodels con link logarítmico. Es significativamente más
   lento que TreeExplainer, por lo que se limita a una muestra reducida
   (200 personas evaluadas, 100 de background) — puede tardar 1-3
   minutos en ejecutarse, es esperado.

SHAP global: importancia promedio de cada feature sobre el set evaluado.
SHAP local: explicación individual — qué features empujaron la
predicción de una persona específica hacia arriba o abajo respecto al
valor base del modelo.
"""

import pandas as pd
import numpy as np
import shap
import warnings
from pathlib import Path

from src.analytics.cost_drivers import FEATURES
from src.models.claim_sev import cargar_modelo_severidad, predecir_severidad
from src.segmentation.risk_detection import cargar_modelo_lightgbm

ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# 1. SHAP PARA LIGHTGBM (RIESGO) — TreeExplainer
# =========================================================
def shap_lightgbm(df, n_muestra=2000, random_state=42):
    """
    SHAP TreeExplainer sobre el modelo LightGBM de alto riesgo.
    Rápido y exacto; se limita a n_muestra por practicidad, no por
    necesidad del método.
    """
    modelo, _ = cargar_modelo_lightgbm()

    df_model = df[FEATURES].dropna().reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    idx_muestra = rng.choice(len(df_model), size=min(n_muestra, len(df_model)), replace=False)
    X_muestra = df_model.iloc[idx_muestra].reset_index(drop=True)

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_muestra)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # clase positiva (alto_coste=1)

    print(f"[SHAP] LightGBM — calculado sobre {len(X_muestra):,} personas")

    return explainer, shap_values, X_muestra


def shap_global_lightgbm(shap_values, X_muestra):
    """
    Importancia global — promedio del valor absoluto de SHAP por
    feature.
    """
    importancia = pd.DataFrame({
        "variable": X_muestra.columns,
        "shap_importancia_media": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_importancia_media", ascending=False).reset_index(drop=True)

    print("\nSHAP global — LightGBM (alto riesgo), top 10:")
    print(importancia.head(10).to_string(index=False))

    return importancia


def shap_local_lightgbm(explainer, shap_values, X_muestra, idx_persona=0):
    """
    Explicación local — desglosa la predicción de una persona
    específica.
    """
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    fila_shap = shap_values[idx_persona]
    fila_valores = X_muestra.iloc[idx_persona]

    tabla = pd.DataFrame({
        "variable": X_muestra.columns,
        "valor_persona": fila_valores.values,
        "shap_value": fila_shap,
    })
    tabla["abs_shap"] = tabla["shap_value"].abs()
    tabla = tabla.sort_values("abs_shap", ascending=False).drop(columns="abs_shap").reset_index(drop=True)

    print(f"\nSHAP local — persona índice {idx_persona} (LightGBM):")
    print(f"  Valor base (expected_value): {base_value:.4f}")
    print(f"  Suma SHAP + base = predicción: {base_value + fila_shap.sum():.4f}")
    print(tabla.head(10).to_string(index=False))
    print("\n→ shap_value > 0: la variable empuja hacia MAYOR probabilidad de alto riesgo")
    print("→ shap_value < 0: la variable empuja hacia MENOR probabilidad de alto riesgo")

    return tabla


# =========================================================
# 2. SHAP PARA GLM GAMMA (SEVERIDAD) — KernelExplainer
# =========================================================
def shap_severidad_gamma(df, n_muestra=200, n_background=100, random_state=42):
    """
    SHAP KernelExplainer (model-agnostic) sobre el modelo Gamma de
    severidad. Limitado a muestra pequeña por costo computacional.
    """
    modelo_sev, _ = cargar_modelo_severidad()

    df_model = df[FEATURES + ["gasto_total_anual"]].dropna()
    df_model = df_model[df_model["gasto_total_anual"] > 0][FEATURES].reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    idx_background = rng.choice(len(df_model), size=min(n_background, len(df_model)), replace=False)
    idx_evaluar = rng.choice(len(df_model), size=min(n_muestra, len(df_model)), replace=False)

    X_background = df_model.iloc[idx_background].reset_index(drop=True)
    X_evaluar = df_model.iloc[idx_evaluar].reset_index(drop=True)

    def f_predict(X_array):
        X_df = pd.DataFrame(X_array, columns=FEATURES)
        return predecir_severidad(X_df, modelo=modelo_sev).values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(f_predict, X_background)
        shap_values = explainer.shap_values(X_evaluar, nsamples=100)

    print(f"[SHAP] GLM Gamma (severidad) — calculado sobre {len(X_evaluar):,} personas "
          f"(background: {len(X_background):,})")

    return explainer, shap_values, X_evaluar


def shap_global_severidad(shap_values, X_evaluar):
    importancia = pd.DataFrame({
        "variable": X_evaluar.columns,
        "shap_importancia_media": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_importancia_media", ascending=False).reset_index(drop=True)

    print("\nSHAP global — GLM Gamma (severidad), top 10:")
    print(importancia.head(10).to_string(index=False))

    return importancia


def shap_local_severidad(explainer, shap_values, X_evaluar, idx_persona=0):
    """
    Explicación local para severidad — misma lógica que
    shap_local_lightgbm, adaptada a un solo array de shap_values
    (KernelExplainer con función de salida escalar no devuelve lista).
    """
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]

    fila_shap = shap_values[idx_persona]
    fila_valores = X_evaluar.iloc[idx_persona]

    tabla = pd.DataFrame({
        "variable": X_evaluar.columns,
        "valor_persona": fila_valores.values,
        "shap_value": fila_shap,
    })
    tabla["abs_shap"] = tabla["shap_value"].abs()
    tabla = tabla.sort_values("abs_shap", ascending=False).drop(columns="abs_shap").reset_index(drop=True)

    print(f"\nSHAP local — persona índice {idx_persona} (GLM Gamma severidad):")
    print(f"  Valor base (expected_value): ${base_value:,.1f}")
    print(f"  Suma SHAP + base = predicción: ${base_value + fila_shap.sum():,.1f}")
    print(tabla.head(10).to_string(index=False))
    print("\n→ shap_value > 0: la variable empuja el gasto esperado hacia ARRIBA")
    print("→ shap_value < 0: la variable empuja el gasto esperado hacia ABAJO")

    return tabla


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> SHAP — LightGBM (alto riesgo)...")
    explainer_lgbm, shap_values_lgbm, X_muestra_lgbm = shap_lightgbm(df)
    importancia_global_lgbm = shap_global_lightgbm(shap_values_lgbm, X_muestra_lgbm)
    shap_local_lightgbm(explainer_lgbm, shap_values_lgbm, X_muestra_lgbm, idx_persona=0)

    print("\n>>> SHAP — GLM Gamma (severidad)... (puede tardar 1-3 minutos)")
    explainer_sev, shap_values_sev, X_evaluar_sev = shap_severidad_gamma(df)
    importancia_global_sev = shap_global_severidad(shap_values_sev, X_evaluar_sev)
    shap_local_severidad(explainer_sev, shap_values_sev, X_evaluar_sev, idx_persona=0)