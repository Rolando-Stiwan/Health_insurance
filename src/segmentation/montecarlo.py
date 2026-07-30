"""
montecarlo.py
----------------
Simulación Monte Carlo para VaR, CVaR y capital económico.
Segundo módulo de la Parte 4 (Explainability & business insights).

Metodología (misma familia conceptual que CreditRiskLab, adaptada al
modelo de dos partes de este proyecto):
1. Por cada persona del portafolio, se dibuja:
   a) Ocurrencia de gasto ~ Bernoulli(prob_gasto), del modelo logit
      (pricing_engine.py).
   b) Si ocurre gasto, severidad ~ Gamma(shape, scale), parametrizada
      con la media predicha por claim_sev.py y el parámetro de
      dispersión real estimado por el GLM Gamma (model.scale).
2. Se suma el gasto de todas las personas del portafolio en cada
   simulación → distribución de pérdida agregada del portafolio.
3. Se repite N veces (default 10,000) para construir la distribución.
4. Se calculan: pérdida esperada, VaR 99%, CVaR 99%, capital económico
   (VaR - pérdida esperada, el "buffer" de pérdida inesperada).

Supuesto de independencia: las personas se simulan independientemente
entre sí (sin correlación sistemática) — apropiado como línea base de
riesgo idiosincrático. Escenarios correlacionados (ej. pandemia) se
tratan por separado en stress_test.py.

Portafolio = la muestra real de personas con datos completos (~14,728),
no la población expandida por peso_muestral — enfoque estándar para
capital económico de un portafolio de asegurados concreto.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.analytics.cost_drivers import FEATURES
from src.models.pricing_engine import (
    cargar_modelo_probabilidad, cargar_modelo_severidad,
    predecir_probabilidad_gasto, predecir_severidad
)

ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# 1. PREPARAR PARÁMETROS DE SIMULACIÓN POR PERSONA
# =========================================================
def preparar_parametros_simulacion(df):
    """
    Calcula, para cada persona, prob_gasto (Bernoulli) y los parámetros
    de la Gamma de severidad (shape, scale) a partir de la media
    predicha y la dispersión real del GLM Gamma entrenado.
    """
    modelo_prob, _ = cargar_modelo_probabilidad()
    modelo_sev, _ = cargar_modelo_severidad()

    df_model = df[FEATURES + ["persona_id"]].dropna().reset_index(drop=True)

    prob_gasto = predecir_probabilidad_gasto(df_model, modelo=modelo_prob)
    severidad_media = predecir_severidad(df_model, modelo=modelo_sev)

    dispersion = modelo_sev.scale  # scale del GLM Gamma (Pearson-based)
    shape_gamma = 1.0 / dispersion  # k = 1/dispersión
    scale_gamma = severidad_media * dispersion  # theta = mu * dispersión (para que k*theta = mu)

    df_model["prob_gasto"] = prob_gasto
    df_model["severidad_media"] = severidad_media
    df_model["shape_gamma"] = shape_gamma
    df_model["scale_gamma"] = scale_gamma

    print(f"[montecarlo] Parámetros preparados para {len(df_model):,} personas")
    print(f"[montecarlo] Dispersión del GLM Gamma (model.scale): {dispersion:.4f}")
    print(f"[montecarlo] Shape de la Gamma de severidad: {shape_gamma:.4f}")

    return df_model


# =========================================================
# 2. SIMULACIÓN MONTE CARLO — PÉRDIDA AGREGADA DEL PORTAFOLIO
# =========================================================
def simular_perdida_portafolio(df_params, n_simulaciones=10000, random_state=42):
    """
    Simula n_simulaciones escenarios de pérdida agregada del portafolio.
    Vectorizado: dibuja Bernoulli y Gamma para las N personas x
    n_simulaciones de una sola vez, en bloques para no agotar memoria.
    """
    rng = np.random.default_rng(random_state)
    n_personas = len(df_params)

    prob_gasto = df_params["prob_gasto"].values
    shape_gamma = df_params["shape_gamma"].values
    scale_gamma = df_params["scale_gamma"].values

    perdidas_agregadas = np.empty(n_simulaciones)

    bloque = 500  # simulaciones por bloque, para controlar uso de memoria
    for inicio in range(0, n_simulaciones, bloque):
        fin = min(inicio + bloque, n_simulaciones)
        n_bloque = fin - inicio

        ocurrencia = rng.random((n_bloque, n_personas)) < prob_gasto
        severidad = rng.gamma(shape_gamma, scale_gamma, size=(n_bloque, n_personas))

        gasto_simulado = ocurrencia * severidad
        perdidas_agregadas[inicio:fin] = gasto_simulado.sum(axis=1)

    print(f"\n[montecarlo] {n_simulaciones:,} simulaciones completadas")
    print(f"[montecarlo] Pérdida agregada — media: ${perdidas_agregadas.mean():,.0f}")
    print(f"[montecarlo] Pérdida agregada — desviación estándar: ${perdidas_agregadas.std():,.0f}")

    return perdidas_agregadas


# =========================================================
# 3. VaR, CVaR Y CAPITAL ECONÓMICO
# =========================================================
def calcular_var_cvar(perdidas_agregadas, niveles_confianza=(0.95, 0.99)):
    """
    Calcula VaR y CVaR (Expected Shortfall) a los niveles de confianza
    dados, más el capital económico como buffer sobre la pérdida
    esperada.
    """
    perdida_esperada = perdidas_agregadas.mean()

    resultados = []
    for nivel in niveles_confianza:
        var = np.percentile(perdidas_agregadas, nivel * 100)
        cvar = perdidas_agregadas[perdidas_agregadas >= var].mean()
        capital_economico = var - perdida_esperada

        resultados.append({
            "nivel_confianza": f"{nivel*100:.0f}%",
            "VaR": round(var, 0),
            "CVaR": round(cvar, 0),
            "capital_economico": round(capital_economico, 0),
        })

    tabla = pd.DataFrame(resultados)

    print(f"\nPérdida esperada del portafolio: ${perdida_esperada:,.0f}")
    print("\nVaR, CVaR y capital económico:")
    print(tabla.to_string(index=False))
    print("\n→ VaR: pérdida máxima esperada al nivel de confianza dado")
    print("→ CVaR: pérdida media EN LOS ESCENARIOS que superan el VaR (más conservador)")
    print("→ Capital económico: buffer necesario sobre la pérdida esperada (VaR - pérdida esperada)")

    return tabla, perdida_esperada


# =========================================================
# 4. COMPARACIÓN CON SUMA DE PRIMAS PURAS (CHEQUEO DE SANIDAD)
# =========================================================
def comparar_con_prima_pura(df_params, perdida_esperada):
    """
    Chequeo de sanidad: la pérdida esperada de la simulación debe
    coincidir aproximadamente con la suma de primas puras individuales
    (prob_gasto x severidad_media), ya que E[simulación] converge a
    esa cantidad por la ley de los grandes números.
    """
    suma_prima_pura = (df_params["prob_gasto"] * df_params["severidad_media"]).sum()
    diferencia_pct = (perdida_esperada - suma_prima_pura) / suma_prima_pura * 100

    print(f"\nChequeo de sanidad — pérdida esperada simulada vs suma de primas puras:")
    print(f"  Suma de primas puras (analítico): ${suma_prima_pura:,.0f}")
    print(f"  Pérdida esperada (simulación): ${perdida_esperada:,.0f}")
    print(f"  Diferencia: {diferencia_pct:+.2f}%")
    print("→ Diferencia pequeña (<2-3%) confirma que la simulación converge correctamente")

    return suma_prima_pura, diferencia_pct


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Preparando parámetros de simulación...")
    df_params = preparar_parametros_simulacion(df)

    print("\n>>> Ejecutando simulación Monte Carlo (10,000 escenarios)...")
    perdidas = simular_perdida_portafolio(df_params)

    print("\n>>> Calculando VaR, CVaR y capital económico...")
    tabla_riesgo, perdida_esperada = calcular_var_cvar(perdidas)

    print("\n>>> Chequeo de sanidad...")
    comparar_con_prima_pura(df_params, perdida_esperada)