"""
claim_sev.py
--------------
Modelo de severidad de siniestros — GLM Gamma.
Segundo módulo de la Parte 2 (Predictive modeling).

Reutiliza la distribución ya seleccionada formalmente en
diagnostico_distribucion.py (Gamma, ver decisiones_tecnicas.md) y las
mismas FEATURES de cost_drivers.py, pero con un enfoque distinto:
mientras cost_drivers.py es para interpretabilidad y análisis de
drivers (Parte 1), este módulo produce el artefacto de modelo
persistido que pricing_engine.py consume para generar predicciones
de severidad (Parte 2).

Arquitectura: entrenamiento y predicción están separados.
- entrenar_y_guardar_severidad(): se corre explícitamente para
  (re)entrenar. Guarda el modelo en models_artifacts/ vía joblib,
  junto con metadata (fecha, métricas, features) en JSON.
- cargar_modelo_severidad() / predecir_severidad(): usadas en
  producción (pricing_engine.py, futuro endpoint FastAPI). Cargan el
  artefacto ya entrenado, sin reentrenar — evita reentrenar el modelo
  en cada llamada, estándar de separación training/inference en MLOps.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.analytics.cost_drivers import FEATURES

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "models_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "severity_gamma.joblib"
META_PATH = ARTIFACTS_DIR / "severity_gamma_meta.json"


# =========================================================
# 1. ENTRENAMIENTO Y PERSISTENCIA
# =========================================================
def entrenar_y_guardar_severidad(df, col="gasto_total_anual", peso="peso_muestral"):
    """
    Entrena GLM Gamma sobre split 80/20 (random_state=42, consistente
    con cost_drivers.py y diagnostico_distribucion.py). Guarda el
    modelo entrenado y su metadata en models_artifacts/.
    """
    df_model = df[FEATURES + [col, peso]].dropna()
    df_model = df_model[df_model[col] > 0].reset_index(drop=True)

    X = sm.add_constant(df_model[FEATURES])
    y = df_model[col].astype(float).values
    w = df_model[peso].values
    w = w / w.mean()

    idx = df_model.index.values
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    w_train = w[idx_train]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = sm.GLM(
            y_train, X_train,
            family=sm.families.Gamma(link=sm.families.links.Log()),
            freq_weights=w_train
        ).fit()

    pred = modelo.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    mae = np.mean(np.abs(y_test - pred))

    print(f"[claim_sev] GLM Gamma entrenado — RMSE={rmse:,.1f}  MAE={mae:,.1f}")

    # Persistir modelo
    joblib.dump(modelo, MODEL_PATH)

    # Persistir metadata
    metadata = {
        "fecha_entrenamiento": datetime.now().isoformat(),
        "distribucion": "Gamma",
        "link_function": "log",
        "features": FEATURES,
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "rmse_test": round(rmse, 1),
        "mae_test": round(mae, 1),
        "random_state": 42,
        "nota": "Distribución seleccionada formalmente en diagnostico_distribucion.py",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[claim_sev] Modelo guardado: {MODEL_PATH.name}")
    print(f"[claim_sev] Metadata guardada: {META_PATH.name}")

    return modelo, metadata


# =========================================================
# 2. CARGA DEL MODELO PERSISTIDO
# =========================================================
def cargar_modelo_severidad():
    """
    Carga el modelo de severidad ya entrenado desde disco.
    Usar en pricing_engine.py y futuros endpoints — no reentrena.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe modelo entrenado en {MODEL_PATH}. "
            f"Correr entrenar_y_guardar_severidad() primero."
        )

    modelo = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return modelo, metadata


# =========================================================
# 3. PREDICCIÓN DE SEVERIDAD
# =========================================================
def predecir_severidad(df_nuevo, modelo=None):
    """
    Genera predicción de severidad (gasto esperado dado que hay gasto)
    para nuevas observaciones. Si no se pasa modelo, lo carga desde disco.

    df_nuevo debe tener las columnas de FEATURES (no necesita
    gasto_total_anual ni peso_muestral, solo para predecir).
    """
    if modelo is None:
        modelo, _ = cargar_modelo_severidad()

    faltantes = [f for f in FEATURES if f not in df_nuevo.columns]
    if faltantes:
        raise ValueError(f"Faltan features requeridas para predecir: {faltantes}")

    X_nuevo = sm.add_constant(df_nuevo[FEATURES], has_constant="add")
    pred = modelo.predict(X_nuevo)

    return pred


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Entrenando y guardando modelo de severidad...")
    modelo, metadata = entrenar_y_guardar_severidad(df)

    print("\n>>> Verificando carga desde disco...")
    modelo_cargado, metadata_cargada = cargar_modelo_severidad()
    print(f"  Modelo cargado — entrenado el {metadata_cargada['fecha_entrenamiento']}")
    print(f"  RMSE test: {metadata_cargada['rmse_test']}")

    print("\n>>> Probando predicción sobre una muestra pequeña...")
    df_muestra = df[FEATURES].dropna().head(5)
    pred_muestra = predecir_severidad(df_muestra, modelo=modelo_cargado)
    print(pred_muestra.values)