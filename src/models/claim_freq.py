"""
claim_freq.py
--------------
Modelo de frecuencia de siniestros — GLM Poisson vs Binomial Negativa.
Primer módulo de la Parte 2 (Predictive modeling).

Target: n_eventos_utilizacion = visitas_ambulatorias + visitas_outpatient +
        visitas_urgencias + noches_hospital

Decisión de diseño: se excluye total_recetas del conteo de frecuencia.
Una receta es un evento de dispensación farmacéutica, no un encuentro
médico — incluirla distorsionaría la frecuencia, ya que el promedio de
recetas/año (~9) es mayor que el de visitas (~7) y dominaría el conteo
combinado sin representar frecuencia de siniestro en sentido actuarial.

Todas las observaciones se incluyen, incluyendo frecuencia=0 — es una
observación válida en un modelo de conteo.

Metodología de selección de distribución:
Poisson asume varianza = media. Se diagnostica sobredispersión antes de
aceptar Poisson como final; si hay evidencia fuerte (ratio var/media >> 1),
se compara formalmente contra Binomial Negativa (alpha estimado vía
regresión auxiliar ponderada de Cameron-Trivedi) y se decide por RMSE
fuera de muestra, mismo criterio que en diagnostico_distribucion.py.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.analytics.cost_drivers import FEATURES
import joblib
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

COLS_EVENTO = [
    "visitas_ambulatorias", "visitas_outpatient",
    "visitas_urgencias", "noches_hospital",
]


# =========================================================
# 1. CONSTRUCCIÓN DEL TARGET DE FRECUENCIA
# =========================================================
def construir_target_frecuencia(df):
    """
    Suma las columnas de utilización que representan encuentros médicos
    (excluye recetas, ver docstring del módulo).
    """
    faltantes = [c for c in COLS_EVENTO if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas de utilización faltantes: {faltantes}")

    df = df.copy()
    df["n_eventos_utilizacion"] = df[COLS_EVENTO].fillna(0).sum(axis=1)
    df["n_eventos_utilizacion"] = df["n_eventos_utilizacion"].round().astype(int)

    return df


# =========================================================
# 2. DIAGNÓSTICO DE MISSING-NOT-AT-RANDOM EN FEATURES
# =========================================================
def diagnostico_missing(df, col="n_eventos_utilizacion"):
    """
    Compara la frecuencia media entre personas con dato completo vs
    incompleto en cada feature, para detectar si el dropna() introduce
    sesgo de selección (missing-not-at-random) antes de entrenar.
    """
    resultados = []
    for feat in FEATURES:
        falta = df[feat].isna()
        if falta.sum() == 0:
            continue

        media_falta = df.loc[falta, col].mean()
        media_completo = df.loc[~falta, col].mean()

        resultados.append({
            "feature": feat,
            "n_falta": int(falta.sum()),
            "pct_falta": round(falta.mean() * 100, 2),
            "freq_media_si_falta": round(media_falta, 2),
            "freq_media_si_completo": round(media_completo, 2),
            "diferencia_pct": round(
                (media_completo - media_falta) / media_falta * 100, 1
            ) if media_falta > 0 else np.nan,
        })

    tabla = pd.DataFrame(resultados).sort_values("n_falta", ascending=False).reset_index(drop=True)

    print("\nDiagnóstico de missing-not-at-random (por feature):")
    if len(tabla) == 0:
        print("  Sin valores faltantes en ninguna feature")
    else:
        print(tabla.to_string(index=False))
        print("\n→ Si freq_media_si_completo es mucho mayor que freq_media_si_falta,")
        print("  el dropna() está sesgando la muestra hacia personas de mayor utilización")

    return tabla


# =========================================================
# 3. ENTRENAMIENTO GLM POISSON
# =========================================================
def entrenar_glm_poisson(df, col="n_eventos_utilizacion", peso="peso_muestral"):
    """
    Entrena GLM Poisson con pesos muestrales. Incluye todas las
    observaciones (frecuencia=0 es válida).
    Evalúa fuera de muestra (80/20, random_state=42).
    """
    df_model = df[FEATURES + [col, peso]].dropna().reset_index(drop=True)

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
            family=sm.families.Poisson(link=sm.families.links.Log()),
            freq_weights=w_train
        ).fit()

    pred = modelo.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    mae = np.mean(np.abs(y_test - pred))

    print(f"[GLM] Poisson         — RMSE={rmse:,.3f}  MAE={mae:,.3f}  AIC={modelo.aic:,.1f}")

    return modelo, {"rmse": round(rmse, 3), "mae": round(mae, 3), "aic": round(modelo.aic, 1)}


# =========================================================
# 4. DIAGNÓSTICO DE SOBREDISPERSIÓN
# =========================================================
def diagnostico_sobredispersion(modelo, df, col="n_eventos_utilizacion", peso="peso_muestral"):
    """
    Poisson asume media = varianza. Reporta ratio varianza/media y
    estadístico de dispersión de Pearson como diagnóstico estándar.
    """
    df_model = df[FEATURES + [col, peso]].dropna().reset_index(drop=True)
    y = df_model[col].astype(float).values

    media = y.mean()
    varianza = y.var()
    ratio_var_media = varianza / media

    pearson_chi2 = modelo.pearson_chi2
    df_resid = modelo.df_resid
    dispersion = pearson_chi2 / df_resid

    print(f"\nDiagnóstico de sobredispersión:")
    print(f"  Media observada: {media:.3f}")
    print(f"  Varianza observada: {varianza:.3f}")
    print(f"  Ratio varianza/media: {ratio_var_media:.2f}")
    print(f"  Estadístico de dispersión (Pearson chi2 / df_resid): {dispersion:.2f}")

    if dispersion > 1.5:
        print("  → Dispersión > 1.5: evidencia de sobredispersión")
        print("  → Se compara formalmente contra Binomial Negativa")
    else:
        print("  → Dispersión cercana a 1: supuesto Poisson razonable")

    return {"ratio_var_media": round(ratio_var_media, 2), "dispersion": round(dispersion, 2)}


# =========================================================
# 5. ESTIMACIÓN DE ALPHA Y ENTRENAMIENTO BINOMIAL NEGATIVA
# =========================================================
def estimar_alpha_binomial_negativa(modelo_poisson, X_train, y_train, w_train):
    """
    Estima el parámetro de dispersión alpha vía regresión auxiliar
    ponderada de Cameron-Trivedi: regresiona ((y-mu)^2 - y) / mu contra
    mu (sin intercepto), el coeficiente resultante es alpha.
    Método estándar cuando GLM NB de statsmodels requiere alpha fijo.
    """
    mu = modelo_poisson.predict(X_train)
    aux_y = ((y_train - mu) ** 2 - y_train) / mu
    aux_X = mu

    aux_modelo = sm.WLS(aux_y, aux_X, weights=w_train).fit()
    alpha = aux_modelo.params[0]

    alpha = max(alpha, 1e-4)  # alpha debe ser positivo
    print(f"\n[NB] Alpha estimado (Cameron-Trivedi): {alpha:.4f}")

    return alpha


def entrenar_glm_binomial_negativa(df, col="n_eventos_utilizacion", peso="peso_muestral"):
    """
    Entrena GLM Binomial Negativa con alpha estimado y pesos muestrales.
    Mismo split 80/20, random_state=42, para comparación directa con
    entrenar_glm_poisson().
    """
    df_model = df[FEATURES + [col, peso]].dropna().reset_index(drop=True)

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
        modelo_poisson_aux = sm.GLM(
            y_train, X_train,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            freq_weights=w_train
        ).fit()

        alpha = estimar_alpha_binomial_negativa(modelo_poisson_aux, X_train, y_train, w_train)

        modelo = sm.GLM(
            y_train, X_train,
            family=sm.families.NegativeBinomial(alpha=alpha, link=sm.families.links.Log()),
            freq_weights=w_train
        ).fit()

    pred = modelo.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    mae = np.mean(np.abs(y_test - pred))

    print(f"[GLM] Binomial Neg.    — RMSE={rmse:,.3f}  MAE={mae:,.3f}  AIC={modelo.aic:,.1f}")

    return modelo, {"rmse": round(rmse, 3), "mae": round(mae, 3), "aic": round(modelo.aic, 1), "alpha": round(alpha, 4)}


# =========================================================
# 6. COMPARACIÓN FORMAL — DECISIÓN POR RMSE
# =========================================================
def comparar_frecuencia(df, col="n_eventos_utilizacion", peso="peso_muestral"):
    """
    Compara Poisson vs Binomial Negativa fuera de muestra.
    Criterio de decisión: RMSE (consistente con el criterio ya usado
    en diagnostico_distribucion.py para severidad).
    """
    print("\nComparación formal Poisson vs Binomial Negativa:")
    modelo_poisson, metricas_poisson = entrenar_glm_poisson(df, col, peso)
    modelo_nb, metricas_nb = entrenar_glm_binomial_negativa(df, col, peso)

    tabla = pd.DataFrame([
        {"modelo": "Poisson", **metricas_poisson},
        {"modelo": "Binomial Negativa", **{k: v for k, v in metricas_nb.items() if k != "alpha"}},
    ])
    print("\n" + tabla.to_string(index=False))

    ganador = tabla.loc[tabla["rmse"].idxmin(), "modelo"]
    print(f"\n→ Mejor modelo fuera de muestra: {ganador}")
    print("→ Documentar resultado en decisiones_tecnicas.md")

    modelos = {"Poisson": modelo_poisson, "Binomial Negativa": modelo_nb}
    return modelos, tabla, ganador


# =========================================================
# 7. RANKING DE DRIVERS DE FRECUENCIA
# =========================================================
def ranking_drivers_frecuencia(modelo, nombre_modelo="Binomial Negativa"):
    """
    Extrae coeficientes del GLM y genera ranking de drivers, mismo
    formato que ranking_drivers() en cost_drivers.py.
    """
    summary = modelo.summary2().tables[1]
    p_col = "P>|z|" if "P>|z|" in summary.columns else "P>|t|"

    drivers = pd.DataFrame({
        "variable": summary.index,
        "coeficiente": summary["Coef."].round(4),
        "exp_coef": np.exp(summary["Coef."]).round(4),
        "p_valor": summary[p_col].round(4),
        "significativo": summary[p_col] < 0.05,
    })

    drivers = drivers[drivers["variable"] != "const"]
    drivers["impacto_abs"] = drivers["coeficiente"].abs()
    drivers = drivers.sort_values("impacto_abs", ascending=False).reset_index(drop=True)

    print(f"\nRanking de drivers — Frecuencia ({nombre_modelo}):")
    print(f"{'Variable':<30} {'Coef':>8} {'exp(Coef)':>10} {'p-valor':>8} {'Sig':>5}")
    print("-" * 65)
    for _, row in drivers.iterrows():
        sig = "✓" if row["significativo"] else " "
        print(f"{row['variable']:<30} {row['coeficiente']:>8.4f} {row['exp_coef']:>10.4f} {row['p_valor']:>8.4f} {sig:>5}")

    print("\n→ exp(coef) > 1 = aumenta la frecuencia esperada de eventos")
    print("→ exp(coef) < 1 = reduce la frecuencia esperada de eventos")
    print("→ ✓ = estadísticamente significativo (p < 0.05)")

    return drivers


# =========================================================
# 8. PERSISTENCIA DEL MODELO GANADOR (Poisson, por RMSE)
# =========================================================

ARTIFACTS_DIR = ROOT / "models_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

FREQ_MODEL_PATH = ARTIFACTS_DIR / "frequency_poisson.joblib"
FREQ_META_PATH = ARTIFACTS_DIR / "frequency_poisson_meta.json"


def entrenar_y_guardar_frecuencia(df, col="n_eventos_utilizacion", peso="peso_muestral"):
    """
    Entrena el modelo ganador (Poisson, seleccionado por RMSE en
    comparar_frecuencia) y lo persiste para uso en pricing_engine.py.
    """
    modelo, metricas = entrenar_glm_poisson(df, col, peso)

    joblib.dump(modelo, FREQ_MODEL_PATH)

    metadata = {
        "fecha_entrenamiento": datetime.now().isoformat(),
        "distribucion": "Poisson",
        "link_function": "log",
        "features": FEATURES,
        "rmse_test": metricas["rmse"],
        "mae_test": metricas["mae"],
        "aic": metricas["aic"],
        "random_state": 42,
        "nota": "Seleccionado sobre Binomial Negativa por RMSE fuera de "
                "muestra (ver decisiones_tecnicas.md). Sobredispersión "
                "conocida y documentada (ratio var/media ~28).",
    }
    with open(FREQ_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[claim_freq] Modelo guardado: {FREQ_MODEL_PATH.name}")
    return modelo, metadata


def cargar_modelo_frecuencia():
    """Carga el modelo de frecuencia ya entrenado desde disco."""
    if not FREQ_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe modelo entrenado en {FREQ_MODEL_PATH}. "
            f"Correr entrenar_y_guardar_frecuencia() primero."
        )
    modelo = joblib.load(FREQ_MODEL_PATH)
    with open(FREQ_META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


def predecir_frecuencia(df_nuevo, modelo=None):
    """Predice frecuencia esperada de eventos para nuevas observaciones."""
    if modelo is None:
        modelo, _ = cargar_modelo_frecuencia()

    faltantes = [f for f in FEATURES if f not in df_nuevo.columns]
    if faltantes:
        raise ValueError(f"Faltan features requeridas para predecir: {faltantes}")

    X_nuevo = sm.add_constant(df_nuevo[FEATURES], has_constant="add")
    return modelo.predict(X_nuevo)

# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Construyendo target de frecuencia...")
    df = construir_target_frecuencia(df)
    print(f"  n_eventos_utilizacion (dataset completo) — media: {df['n_eventos_utilizacion'].mean():.2f}, "
          f"mediana: {df['n_eventos_utilizacion'].median():.0f}, "
          f"% con 0 eventos: {(df['n_eventos_utilizacion'] == 0).mean() * 100:.1f}%")

    print("\n>>> Diagnosticando missing-not-at-random en features...")
    tabla_missing = diagnostico_missing(df)

    print("\n>>> Comparando Poisson vs Binomial Negativa...")
    modelos, tabla_comparacion, ganador = comparar_frecuencia(df)

    print("\n>>> Diagnóstico de sobredispersión (sobre modelo Poisson)...")
    diag = diagnostico_sobredispersion(modelos["Poisson"], df)

    print(f"\n>>> Extrayendo ranking de drivers del modelo ganador ({ganador})...")
    drivers = ranking_drivers_frecuencia(modelos[ganador], nombre_modelo=ganador)

    print("\n>>> Guardando modelo de frecuencia ganador...")
    modelo_final, meta_final = entrenar_y_guardar_frecuencia(df)