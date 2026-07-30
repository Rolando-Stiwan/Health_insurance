"""
pricing_engine.py
--------------------
Motor de pricing: prima pura = P(gasto>0) × severidad esperada,
con credibilidad de Bühlmann calculada bajo dos enfoques paralelos
para comparación (validación de modelo) y un enfoque de producción.

Metodología (modelo de dos partes, definido en decisiones_tecnicas.md):
1. Parte 1: P(gasto>0) — modelo logístico, entrenado aquí mismo.
2. Parte 2: E[gasto | gasto>0] — severidad esperada (claim_sev.py).
3. Prima pura = P(gasto>0) × severidad_esperada.
4. Credibilidad de Bühlmann por segmento (tipo_cobertura), dos enfoques:

   a) CREDIBILIDAD SOBRE EL MODELO (uso en producción):
      mezcla la prima pura promedio del segmento (ya modelada) con la
      prima pura promedio general (también modelada). Resultado
      coherente: una sola cifra, consistente con las predicciones del
      modelo. Esta es la que alimentaría un endpoint /pricing real.

   b) CREDIBILIDAD SOBRE EXPERIENCIA CRUDA (uso en validación/comité):
      mezcla el gasto real promedio del segmento con el gasto real
      promedio general, sin pasar por los modelos. Sirve para
      comparar "qué dice el modelo" vs "qué dice la experiencia cruda"
      — un chequeo de sanidad estándar antes de aprobar una tarifa,
      no se usa para fijar precio final.

Nota: claim_freq.py (conteo de eventos de utilización, GLM Poisson) NO
se usa aquí — ver nota extendida en versión anterior de este docstring.
Representa utilización, no probabilidad de gasto; combinarlo con
severidad_anual_total produce doble conteo.
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
from src.models.claim_sev import cargar_modelo_severidad, predecir_severidad

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "models_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

PROB_MODEL_PATH = ARTIFACTS_DIR / "probabilidad_gasto_logit.joblib"
PROB_META_PATH = ARTIFACTS_DIR / "probabilidad_gasto_logit_meta.json"

SEGMENTO_CREDIBILIDAD = "tipo_cobertura"


# =========================================================
# 1. MODELO LOGÍSTICO — P(gasto > 0)
# =========================================================
def entrenar_y_guardar_probabilidad_gasto(df, col="gasto_total_anual", peso="peso_muestral"):
    """
    Entrena GLM Binomial (logit) para P(gasto>0), Parte 1 del modelo
    de dos partes ya definido en decisiones_tecnicas.md.
    """
    df_model = df[FEATURES + [col, peso]].dropna().reset_index(drop=True)

    X = sm.add_constant(df_model[FEATURES])
    y = (df_model[col] > 0).astype(float).values
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
            family=sm.families.Binomial(link=sm.families.links.Logit()),
            freq_weights=w_train
        ).fit()

    pred_prob = modelo.predict(X_test)
    pred_clase = (pred_prob >= 0.5).astype(int)
    accuracy = np.mean(pred_clase == y_test)
    brier = np.mean((pred_prob - y_test) ** 2)

    print(f"[pricing] Logit P(gasto>0) — Accuracy={accuracy:.3f}  Brier={brier:.4f}  AIC={modelo.aic:,.1f}")

    joblib.dump(modelo, PROB_MODEL_PATH)
    metadata = {
        "fecha_entrenamiento": datetime.now().isoformat(),
        "modelo": "Logit (Binomial)",
        "features": FEATURES,
        "accuracy_test": round(accuracy, 3),
        "brier_test": round(brier, 4),
        "aic": round(modelo.aic, 1),
        "random_state": 42,
    }
    with open(PROB_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[pricing] Modelo guardado: {PROB_MODEL_PATH.name}")
    return modelo, metadata


def cargar_modelo_probabilidad():
    if not PROB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe modelo entrenado en {PROB_MODEL_PATH}. "
            f"Correr entrenar_y_guardar_probabilidad_gasto() primero."
        )
    modelo = joblib.load(PROB_MODEL_PATH)
    with open(PROB_META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


def predecir_probabilidad_gasto(df_nuevo, modelo=None):
    if modelo is None:
        modelo, _ = cargar_modelo_probabilidad()

    faltantes = [f for f in FEATURES if f not in df_nuevo.columns]
    if faltantes:
        raise ValueError(f"Faltan features requeridas para predecir: {faltantes}")

    X_nuevo = sm.add_constant(df_nuevo[FEATURES], has_constant="add")
    return modelo.predict(X_nuevo)


# =========================================================
# 2. PRIMA PURA INDIVIDUAL — P(gasto>0) × SEVERIDAD
# =========================================================
def calcular_prima_pura(df):
    """
    Prima pura = P(gasto>0) × E[gasto | gasto>0].
    """
    df_model = df[FEATURES + ["persona_id", "peso_muestral", "gasto_total_anual"]].dropna(
        subset=FEATURES
    ).reset_index(drop=True)

    modelo_prob, _ = cargar_modelo_probabilidad()
    modelo_sev, _ = cargar_modelo_severidad()

    prob_hat = predecir_probabilidad_gasto(df_model, modelo=modelo_prob)
    sev_hat = predecir_severidad(df_model, modelo=modelo_sev)

    df_model["prob_gasto"] = prob_hat
    df_model["severidad_esperada"] = sev_hat
    df_model["prima_pura"] = prob_hat * sev_hat

    print(f"[pricing] Prima pura calculada para {len(df_model):,} personas")
    print(f"[pricing] Prima pura media: ${df_model['prima_pura'].mean():,.1f}")
    print(f"[pricing] Gasto real medio (referencia): ${df_model['gasto_total_anual'].mean():,.1f}")

    return df_model


# =========================================================
# 3. CREDIBILIDAD DE BÜHLMANN — DOS ENFOQUES PARALELOS
# =========================================================
def calcular_credibilidad_buhlmann(df_prima, segmento=SEGMENTO_CREDIBILIDAD,
                                     peso="peso_muestral", basado_en="modelo"):
    """
    Calcula factor de credibilidad Z por categoría del segmento vía
    método de momentos de Bühlmann-Straub.

    basado_en="modelo": usa prima_pura (predicciones del modelo).
        → Enfoque de PRODUCCIÓN: resultado coherente con el modelo,
          una sola cifra de pricing consistente.
    basado_en="experiencia_cruda": usa gasto_total_anual (dato real).
        → Enfoque de VALIDACIÓN/COMITÉ: compara qué dice la experiencia
          cruda vs el modelo. No se usa para fijar precio final.
    """
    col_valor = "prima_pura" if basado_en == "modelo" else "gasto_total_anual"

    grupos = df_prima.groupby(segmento)

    w_total = df_prima[peso].values
    w_total = w_total / w_total.mean()
    promedio_general = np.average(df_prima[col_valor], weights=w_total)

    resumen_segmentos = []
    for categoria, g in grupos:
        w = g[peso].values
        w = w / w.mean()
        n_i = len(g)
        media_i = np.average(g[col_valor], weights=w)
        varianza_i = np.average((g[col_valor] - media_i) ** 2, weights=w)

        resumen_segmentos.append({
            "categoria": categoria,
            "n": n_i,
            "media_observada": media_i,
            "varianza_observada": varianza_i,
        })

    tabla = pd.DataFrame(resumen_segmentos)

    epv = np.average(tabla["varianza_observada"], weights=(tabla["n"] - 1))

    n_total = tabla["n"].sum()
    r = len(tabla)
    suma_ponderada_desvios = np.sum(tabla["n"] * (tabla["media_observada"] - promedio_general) ** 2)
    correccion = (r - 1) * epv
    denominador = n_total - (tabla["n"] ** 2).sum() / n_total

    vhm = max((suma_ponderada_desvios - correccion) / denominador, 1e-6)
    k = epv / vhm

    tabla["Z_credibilidad"] = tabla["n"] / (tabla["n"] + k)
    tabla["prima_credibilidad"] = (
        tabla["Z_credibilidad"] * tabla["media_observada"] +
        (1 - tabla["Z_credibilidad"]) * promedio_general
    )

    etiqueta = "MODELO (producción)" if basado_en == "modelo" else "EXPERIENCIA CRUDA (validación)"
    print(f"\nCredibilidad de Bühlmann por {segmento} — base: {etiqueta}")
    print(f"  Promedio general: ${promedio_general:,.1f}")
    print(f"  k (parámetro de Bühlmann): {k:,.1f}")
    print(tabla[["categoria", "n", "media_observada", "Z_credibilidad", "prima_credibilidad"]]
          .round(3).to_string(index=False))

    return tabla, {"promedio_general": promedio_general, "epv": epv, "vhm": vhm, "k": k, "basado_en": basado_en}


# =========================================================
# 4. COMPARACIÓN MODELO vs EXPERIENCIA CRUDA (para comité/validación)
# =========================================================
def comparar_enfoques_credibilidad(tabla_modelo, tabla_cruda, segmento=SEGMENTO_CREDIBILIDAD):
    """
    Junta ambos enfoques en una sola tabla para revisión — muestra si
    el modelo y la experiencia cruda coinciden o divergen por segmento.
    """
    comparacion = tabla_modelo[["categoria", "n", "prima_credibilidad"]].rename(
        columns={"prima_credibilidad": "prima_credibilidad_MODELO"}
    ).merge(
        tabla_cruda[["categoria", "prima_credibilidad"]].rename(
            columns={"prima_credibilidad": "prima_credibilidad_EXPERIENCIA_CRUDA"}
        ),
        on="categoria"
    )
    comparacion["diferencia_pct"] = (
        (comparacion["prima_credibilidad_MODELO"] - comparacion["prima_credibilidad_EXPERIENCIA_CRUDA"])
        / comparacion["prima_credibilidad_EXPERIENCIA_CRUDA"] * 100
    ).round(1)

    print(f"\nComparación de enfoques — modelo vs experiencia cruda (por {segmento}):")
    print(comparacion.round(1).to_string(index=False))
    print("\n→ Diferencias grandes indican que el modelo no está bien calibrado para ese segmento")
    print("→ Ver deviation_analysis.py para el diagnóstico de calibración ya documentado")

    return comparacion


# =========================================================
# 5. PRIMA FINAL DE PRODUCCIÓN (solo enfoque modelo)
# =========================================================
def aplicar_prima_produccion(df_prima, tabla_credibilidad_modelo, segmento=SEGMENTO_CREDIBILIDAD):
    """
    Asigna la prima final de producción — únicamente el enfoque basado
    en modelo, consistente con las predicciones de frecuencia/severidad.
    """
    mapa_prima = tabla_credibilidad_modelo.set_index("categoria")["prima_credibilidad"].to_dict()
    mapa_z = tabla_credibilidad_modelo.set_index("categoria")["Z_credibilidad"].to_dict()

    df_prima["prima_final_produccion"] = df_prima[segmento].map(mapa_prima)
    df_prima["Z_credibilidad"] = df_prima[segmento].map(mapa_z)

    print(f"\n[pricing] Prima final de producción asignada por {segmento}")
    print(df_prima[["persona_id", segmento, "prima_pura", "prima_final_produccion", "Z_credibilidad"]]
          .head(10).to_string(index=False))

    return df_prima


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Entrenando y guardando modelo de probabilidad P(gasto>0)...")
    modelo_prob, meta_prob = entrenar_y_guardar_probabilidad_gasto(df)

    print("\n>>> Calculando prima pura (P(gasto>0) × severidad)...")
    df_prima = calcular_prima_pura(df)

    print("\n>>> Credibilidad — enfoque MODELO (producción)...")
    tabla_modelo, params_modelo = calcular_credibilidad_buhlmann(df_prima, basado_en="modelo")

    print("\n>>> Credibilidad — enfoque EXPERIENCIA CRUDA (validación)...")
    tabla_cruda, params_cruda = calcular_credibilidad_buhlmann(df_prima, basado_en="experiencia_cruda")

    print("\n>>> Comparando ambos enfoques...")
    comparacion = comparar_enfoques_credibilidad(tabla_modelo, tabla_cruda)

    print("\n>>> Aplicando prima final de producción...")
    df_final = aplicar_prima_produccion(df_prima, tabla_modelo)