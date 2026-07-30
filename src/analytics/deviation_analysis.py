"""
deviation_analysis.py
-----------------------
Análisis de desviaciones real vs esperado (A/E) por segmento y región.
Usa el modelo GLM Gamma entrenado en cost_drivers.py para generar el
gasto esperado por persona, y lo compara contra el gasto real observado,
agregado por distintos cortes de segmentación.

Metodología:
1. Reentrena el GLM Gamma (mismas FEATURES de cost_drivers.py) sobre el
   split de entrenamiento (80%), y genera predicciones para el dataset
   completo (train+test), para tener gasto esperado por persona.
2. Agrega real vs esperado, ponderado por peso_muestral, por segmento
   (región, tipo_cobertura, categoria_pobreza, salud_general, sin_seguro_anual).
3. Calcula ratio A/E (actual/esperado) con intervalo de confianza vía
   bootstrap ponderado.
4. Flag de desviación significativa: IC 95% del ratio no incluye 1.0.

Nota metodológica importante:
Los segmentos analizados aquí (región, tipo_cobertura, etc.) ya son
FEATURES del modelo Gamma. Por diseño, el GLM minimiza el error global,
no necesariamente el error por subgrupo — así que un ratio A/E distinto
de 1.0 en estos segmentos indica que el modelo no captura bien el efecto
no lineal o de interacción de esa variable, útil para calibración.
Para detectar desviaciones "externas" al modelo (ej. eficiencia de
proveedor, variables no incluidas), se necesitaría un dato que MEPS no
tiene a nivel de proveedor individual — ver limitaciones documentadas.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path
from src.analytics.cost_drivers import FEATURES, entrenar_glm

ROOT = Path(__file__).resolve().parents[2]

SEGMENTOS = [
    "region", "tipo_cobertura", "categoria_pobreza",
    "salud_general", "sin_seguro_anual",
]


# =========================================================
# 1. GENERAR PREDICCIONES DEL MODELO GAMMA
# =========================================================
def generar_predicciones(df, col="gasto_total_anual", peso="peso_muestral"):
    """
    Reentrena el GLM Gamma sobre train (80%, mismo random_state=42 que
    cost_drivers.py) y genera predicciones para el dataset completo
    (train+test), para uso en análisis de desviaciones por segmento.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultados = entrenar_glm(df, col=col, peso=peso)
    modelo_gamma = resultados["Gamma"]["modelo"]

    df_model = df[FEATURES + [col, peso]].dropna()
    df_model = df_model[df_model[col] > 0].reset_index(drop=True)

    X = sm.add_constant(df_model[FEATURES])
    df_model["gasto_esperado"] = modelo_gamma.predict(X)
    df_model["gasto_real"] = df_model[col]

    print(f"\n[deviation] Predicciones generadas para {len(df_model):,} personas")
    return df_model


# =========================================================
# 2. DESVIACIÓN A/E POR SEGMENTO — CON BOOTSTRAP
# =========================================================
def desviacion_por_segmento(df_pred, segmento, peso="peso_muestral", n_bootstrap=500):
    """
    Calcula ratio Actual/Esperado (A/E) ponderado por categoría del
    segmento, con intervalo de confianza vía bootstrap ponderado (500
    remuestreos, seed fija para reproducibilidad).
    """
    rng = np.random.default_rng(42)
    resultados = []

    for valor, g in df_pred.groupby(segmento):
        w = g[peso].values
        w = w / w.mean()
        real_vals = g["gasto_real"].values
        esp_vals = g["gasto_esperado"].values

        real = np.average(real_vals, weights=w)
        esperado = np.average(esp_vals, weights=w)
        ratio_ae = real / esperado

        n = len(g)
        ratios_boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx_boot = rng.integers(0, n, size=n)
            w_boot = w[idx_boot]
            real_boot = np.average(real_vals[idx_boot], weights=w_boot)
            esp_boot = np.average(esp_vals[idx_boot], weights=w_boot)
            ratios_boot[i] = real_boot / esp_boot

        ic_low, ic_high = np.percentile(ratios_boot, [2.5, 97.5])

        resultados.append({
            "segmento": segmento,
            "categoria": valor,
            "n": n,
            "gasto_real": round(real, 1),
            "gasto_esperado": round(esperado, 1),
            "ratio_AE": round(ratio_ae, 3),
            "ic_95_low": round(ic_low, 3),
            "ic_95_high": round(ic_high, 3),
            "desviacion_significativa": not (ic_low <= 1.0 <= ic_high),
        })

    tabla = pd.DataFrame(resultados).sort_values("ratio_AE", ascending=False).reset_index(drop=True)
    return tabla


# =========================================================
# 3. REPORTE COMPLETO — TODOS LOS SEGMENTOS
# =========================================================
def reporte_desviaciones(df_pred, segmentos=SEGMENTOS):
    """
    Corre desviacion_por_segmento() para cada segmento configurado
    e imprime un resumen legible.
    """
    tablas = {}
    for seg in segmentos:
        if seg not in df_pred.columns:
            print(f"[ADVERTENCIA] Segmento '{seg}' no encontrado en el dataset, se omite")
            continue

        tabla = desviacion_por_segmento(df_pred, seg)
        tablas[seg] = tabla

        print(f"\nDesviación Real/Esperado por {seg}:")
        print(tabla.to_string(index=False))

        sig = tabla[tabla["desviacion_significativa"]]
        if len(sig) > 0:
            print(f"→ {len(sig)} categoría(s) con desviación estadísticamente significativa (IC 95% no incluye 1.0)")
        else:
            print("→ Ninguna categoría con desviación significativa")

    print("\n→ ratio_AE > 1 = gasto real superior al esperado (segmento 'infra-tarificado' por el modelo)")
    print("→ ratio_AE < 1 = gasto real inferior al esperado (segmento 'sobre-tarificado' por el modelo)")

    return tablas


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Generando predicciones del modelo Gamma...")
    df_pred = generar_predicciones(df)

    print(">>> Calculando desviaciones por segmento...")
    tablas = reporte_desviaciones(df_pred)