"""
risk_detection.py
--------------------
Detección de alto riesgo — Isolation Forest (no supervisado) +
LightGBM (supervisado). Segundo módulo de la Parte 3 (Risk segmentation).

Target: alto_coste, ya definido en etl.py (top 20% de gasto_total_anual
por año, umbral de cuantil 0.80).

Dos técnicas con propósitos distintos, deliberadamente no intercambiables:
- Isolation Forest: detecta perfiles ATÍPICOS/ANÓMALOS (combinaciones
  raras de características), sin usar el target alto_coste — no
  supervisado. Un perfil anómalo no es necesariamente caro; sirve para
  detectar casos inusuales o posibles problemas de calidad de datos.
- LightGBM: modelo SUPERVISADO que predice directamente la probabilidad
  de alto_coste — la pregunta de negocio real de detección de riesgo.

Se reporta el solapamiento entre ambos enfoques como diagnóstico, no
se espera que coincidan perfectamente.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, classification_report
import lightgbm as lgb

from src.analytics.cost_drivers import FEATURES

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "models_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

LGBM_MODEL_PATH = ARTIFACTS_DIR / "risk_lightgbm.joblib"
LGBM_META_PATH = ARTIFACTS_DIR / "risk_lightgbm_meta.json"

TARGET = "alto_coste"


# =========================================================
# 1. ISOLATION FOREST — DETECCIÓN DE ANOMALÍAS (NO SUPERVISADO)
# =========================================================
def entrenar_isolation_forest(df, contamination=0.20, random_state=42):
    """
    Entrena Isolation Forest sobre FEATURES escaladas. contamination=0.20
    elegido para igualar la prevalencia de alto_coste (top 20%) y así
    poder comparar ambos enfoques directamente — no implica que deban
    coincidir, solo facilita la comparación de proporciones.
    """
    df_model = df[FEATURES + [TARGET]].dropna().reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[FEATURES])

    modelo_if = IsolationForest(
        contamination=contamination, random_state=random_state, n_estimators=200
    )
    anomalia_pred = modelo_if.fit_predict(X_scaled)  # -1 = anómalo, 1 = normal
    df_model["es_anomalia"] = (anomalia_pred == -1).astype(int)

    print(f"[IsolationForest] Entrenado — {df_model['es_anomalia'].sum():,} anomalías "
          f"de {len(df_model):,} personas ({df_model['es_anomalia'].mean()*100:.1f}%)")

    return modelo_if, df_model


def comparar_anomalia_vs_alto_costo(df_model):
    """
    Tabla de contingencia entre es_anomalia (Isolation Forest) y
    alto_coste (target real) — diagnóstico de solapamiento, no se
    espera coincidencia perfecta.
    """
    tabla = pd.crosstab(
        df_model["es_anomalia"], df_model[TARGET],
        rownames=["es_anomalia (Isolation Forest)"], colnames=["alto_coste (real)"]
    )
    print("\nTabla de contingencia — Isolation Forest vs alto_coste real:")
    print(tabla.to_string())

    solapamiento = df_model[(df_model["es_anomalia"] == 1) & (df_model[TARGET] == 1)].shape[0]
    total_anomalias = df_model["es_anomalia"].sum()
    pct_solapamiento = solapamiento / total_anomalias * 100 if total_anomalias > 0 else 0

    print(f"\n→ {pct_solapamiento:.1f}% de las anomalías detectadas también son alto_coste real")
    print("→ Solapamiento parcial es esperado: anomalía ≠ alto costo por definición")
    print("→ Anomalías que NO son alto_coste pueden ser casos inusuales dignos de revisión "
          "(posible error de datos, perfil clínico atípico, etc.)")

    return tabla, pct_solapamiento


# =========================================================
# 2. LIGHTGBM — CLASIFICACIÓN SUPERVISADA DE ALTO RIESGO
# =========================================================
def entrenar_lightgbm_alto_riesgo(df, peso="peso_muestral", random_state=42):
    """
    Entrena LightGBM para predecir alto_coste (top 20% de gasto).
    Split 80/20, mismo random_state=42 del resto del proyecto.
    Dataset desbalanceado (~20/80) — se usa scale_pos_weight y se
    evalúa con AUC-ROC y F1, no con accuracy (engañosa en desbalance).
    """
    df_model = df[FEATURES + [TARGET, peso]].dropna().reset_index(drop=True)

    X = df_model[FEATURES]
    y = df_model[TARGET].astype(int).values
    w = df_model[peso].values
    w = w / w.mean()

    idx = df_model.index.values
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    w_train = w[idx_train]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    modelo = lgb.LGBMClassifier(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        verbosity=-1,
    )
    modelo.fit(X_train, y_train, sample_weight=w_train)

    prob_test = modelo.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)

    auc = roc_auc_score(y_test, prob_test)
    f1 = f1_score(y_test, pred_test)

    print(f"[LightGBM] AUC-ROC={auc:.4f}  F1={f1:.4f}  scale_pos_weight={scale_pos_weight:.2f}")
    print("\nReporte de clasificación (test):")
    print(classification_report(y_test, pred_test, target_names=["No alto costo", "Alto costo"]))

    return modelo, {"auc": round(auc, 4), "f1": round(f1, 4)}, X_test, y_test


def guardar_modelo_lightgbm(modelo, metricas):
    joblib.dump(modelo, LGBM_MODEL_PATH)
    metadata = {
        "fecha_entrenamiento": datetime.now().isoformat(),
        "modelo": "LightGBM Classifier",
        "target": TARGET,
        "features": FEATURES,
        "auc_test": metricas["auc"],
        "f1_test": metricas["f1"],
        "random_state": 42,
    }
    with open(LGBM_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[LightGBM] Modelo guardado: {LGBM_MODEL_PATH.name}")
    return metadata


def cargar_modelo_lightgbm():
    if not LGBM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe modelo entrenado en {LGBM_MODEL_PATH}. "
            f"Correr entrenar_lightgbm_alto_riesgo() + guardar_modelo_lightgbm() primero."
        )
    modelo = joblib.load(LGBM_MODEL_PATH)
    with open(LGBM_META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


# =========================================================
# 3. IMPORTANCIA DE VARIABLES (LightGBM)
# =========================================================
def importancia_variables(modelo):
    """
    Extrae la importancia de variables nativa de LightGBM (ganancia
    total), ranking de qué features más contribuyen a detectar
    alto_coste.
    """
    importancias = pd.DataFrame({
        "variable": modelo.feature_name_,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False).reset_index(drop=True)

    importancias["importancia_pct"] = (
        importancias["importancia"] / importancias["importancia"].sum() * 100
    ).round(1)

    print("\nImportancia de variables (LightGBM) — top 10:")
    print(importancias.head(10).to_string(index=False))

    return importancias



def encontrar_umbral_optimo(modelo, X_test, y_test):
    """
    Busca el umbral de decisión que maximiza F1 sobre el set de test,
    en vez de asumir 0.5 — relevante porque scale_pos_weight ya
    reponderó el entrenamiento, desplazando el punto de corte óptimo.
    """
    prob_test = modelo.predict_proba(X_test)[:, 1]
    precision, recall, umbrales = precision_recall_curve(y_test, prob_test)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    idx_optimo = np.argmax(f1_scores[:-1])  # último punto no tiene umbral asociado
    umbral_optimo = umbrales[idx_optimo]

    print(f"\nUmbral óptimo (máximo F1): {umbral_optimo:.3f}")
    print(f"  Precision en umbral óptimo: {precision[idx_optimo]:.3f}")
    print(f"  Recall en umbral óptimo: {recall[idx_optimo]:.3f}")
    print(f"  F1 en umbral óptimo: {f1_scores[idx_optimo]:.3f}")
    print(f"  (vs umbral 0.5 por defecto: F1={f1_score(y_test, (prob_test>=0.5).astype(int)):.3f})")

    return umbral_optimo, {"precision": precision[idx_optimo], "recall": recall[idx_optimo], "f1": f1_scores[idx_optimo]}

# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Entrenando Isolation Forest (detección de anomalías)...")
    modelo_if, df_anomalias = entrenar_isolation_forest(df)

    print("\n>>> Comparando anomalías vs alto_coste real...")
    tabla_comparacion, pct_solapamiento = comparar_anomalia_vs_alto_costo(df_anomalias)

    print("\n>>> Entrenando LightGBM (clasificación supervisada de alto riesgo)...")
    modelo_lgbm, metricas_lgbm, X_test, y_test = entrenar_lightgbm_alto_riesgo(df)

    print("\n>>> Guardando modelo LightGBM...")
    metadata_lgbm = guardar_modelo_lightgbm(modelo_lgbm, metricas_lgbm)

    print("\n>>> Extrayendo importancia de variables...")
    importancias = importancia_variables(modelo_lgbm)

    print("\n>>> Buscando umbral óptimo de decisión...")
    umbral_optimo, metricas_umbral = encontrar_umbral_optimo(modelo_lgbm, X_test, y_test)