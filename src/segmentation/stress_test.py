"""
stress_test.py
------------------
Pruebas de estrés — escenario pandemia (choque sistémico correlacionado)
y escenario catástrofe regional (choque localizado).
Tercer módulo de la Parte 4 (Explainability & business insights).

Metodología:
1. Escenario PANDEMIA: modelo de un factor (estilo Vasicek/cópula
   gaussiana, la misma técnica usada en CreditRiskLab para riesgo de
   crédito correlacionado). A diferencia de montecarlo.py (personas
   independientes), aquí se introduce un factor sistémico Z común a
   todo el portafolio en cada simulación, que:
   a) Aumenta la probabilidad de gasto de todos correlacionadamente
      (vía cópula gaussiana sobre la Bernoulli).
   b) Aumenta la severidad de todos vía un multiplicador sistémico
      exp(sigma_sistemico * Z).
   Esto captura que, en una pandemia, el riesgo de utilización y coste
   sube para TODOS simultáneamente, no de forma independiente persona
   por persona — el supuesto de independencia de montecarlo.py
   subestimaría la cola de riesgo en un escenario real de pandemia.

2. Escenario CATÁSTROFE REGIONAL: choque de severidad localizado a una
   región específica (ej. huracán, terremoto), multiplicador aplicado
   solo a las personas de esa región — sin correlación sistémica
   global, pero con concentración geográfica del impacto.

3. Comparación tabular: baseline (montecarlo.py) vs cada escenario de
   estrés, en VaR, CVaR y capital económico.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

from src.segmentation.montecarlo import (
    preparar_parametros_simulacion, calcular_var_cvar
)

ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# 1. ESCENARIO PANDEMIA — MODELO DE UN FACTOR (VASICEK/CÓPULA GAUSSIANA)
# =========================================================
def simular_escenario_pandemia(df_params, rho=0.15, multiplicador_severidad_sistemico=0.4,
                                  n_simulaciones=10000, random_state=42):
    """
    Simula pérdida agregada bajo choque sistémico de pandemia, usando
    un modelo de un factor (mismo principio que el modelo de Vasicek
    para riesgo de crédito correlacionado):

    - Factor sistémico Z ~ N(0,1), común a todo el portafolio en cada
      simulación.
    - Ocurrencia de gasto: se corre la probabilidad individual a través
      de una cópula gaussiana con correlación rho al factor Z, en vez
      de un Bernoulli independiente — cuando Z es alto (mal escenario
      sistémico), la probabilidad de gasto de TODOS sube a la vez.
    - Severidad: se multiplica por exp(multiplicador_severidad_sistemico
      * Z) — mismo factor Z, así que cuando el escenario es sistémico-
      mente malo, la severidad también sube para todos a la vez, no
      independientemente.

    rho: correlación al factor sistémico (0=independiente como
    montecarlo.py, 1=todos se mueven idénticamente). rho=0.15 es un
    valor moderado, similar a correlaciones de activos típicas en
    modelos de un factor para carteras de riesgo diversificadas.
    """
    rng = np.random.default_rng(random_state)
    n_personas = len(df_params)

    prob_gasto = df_params["prob_gasto"].values
    shape_gamma = df_params["shape_gamma"].values
    scale_gamma = df_params["scale_gamma"].values

    # Umbral individual en el espacio normal, vía inversa de la normal
    # sobre la probabilidad de gasto (equivalente al "distance to default"
    # de Vasicek, aquí "distancia a generar gasto")
    umbral_normal = stats.norm.ppf(1 - prob_gasto)

    perdidas_agregadas = np.empty(n_simulaciones)

    bloque = 500
    for inicio in range(0, n_simulaciones, bloque):
        fin = min(inicio + bloque, n_simulaciones)
        n_bloque = fin - inicio

        Z = rng.standard_normal(n_bloque)  # factor sistémico, uno por simulación
        epsilon = rng.standard_normal((n_bloque, n_personas))  # idiosincrático

        # Cópula gaussiana de un factor: X_i = sqrt(rho)*Z + sqrt(1-rho)*epsilon_i
        X = np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * epsilon
        ocurrencia = X > umbral_normal[None, :]

        # Severidad con multiplicador sistémico (mismo Z de esta simulación)
        severidad_base = rng.gamma(shape_gamma, scale_gamma, size=(n_bloque, n_personas))
        multiplicador_severidad = np.exp(multiplicador_severidad_sistemico * Z)[:, None]
        severidad = severidad_base * multiplicador_severidad

        gasto_simulado = ocurrencia * severidad
        perdidas_agregadas[inicio:fin] = gasto_simulado.sum(axis=1)

    print(f"\n[stress_test] Escenario PANDEMIA — {n_simulaciones:,} simulaciones "
          f"(rho={rho}, mult. severidad sistémico={multiplicador_severidad_sistemico})")
    print(f"[stress_test] Pérdida agregada — media: ${perdidas_agregadas.mean():,.0f}")
    print(f"[stress_test] Pérdida agregada — desviación estándar: ${perdidas_agregadas.std():,.0f}")

    return perdidas_agregadas


# =========================================================
# 2. ESCENARIO CATÁSTROFE REGIONAL — CHOQUE LOCALIZADO
# =========================================================
def simular_escenario_catastrofe_regional(df_params, region_afectada=3,
                                             multiplicador_severidad=2.5,
                                             n_simulaciones=10000, random_state=42):
    """
    Simula un choque de severidad localizado a una región específica
    (ej. huracán, terremoto) — multiplicador aplicado solo a las
    personas de esa región, resto del portafolio sin cambios respecto
    al baseline. Independencia entre personas (sin factor sistémico),
    a diferencia del escenario pandemia.
    """
    rng = np.random.default_rng(random_state)
    n_personas = len(df_params)

    es_region_afectada = (df_params["region"] == region_afectada).values

    prob_gasto = df_params["prob_gasto"].values
    shape_gamma = df_params["shape_gamma"].values
    scale_gamma = df_params["scale_gamma"].values

    n_afectados = es_region_afectada.sum()
    print(f"\n[stress_test] Escenario CATÁSTROFE — región {region_afectada}, "
          f"{n_afectados:,} personas afectadas ({n_afectados/n_personas*100:.1f}% del portafolio), "
          f"multiplicador severidad={multiplicador_severidad}")

    perdidas_agregadas = np.empty(n_simulaciones)

    bloque = 500
    for inicio in range(0, n_simulaciones, bloque):
        fin = min(inicio + bloque, n_simulaciones)
        n_bloque = fin - inicio

        ocurrencia = rng.random((n_bloque, n_personas)) < prob_gasto
        severidad = rng.gamma(shape_gamma, scale_gamma, size=(n_bloque, n_personas))

        multiplicador = np.where(es_region_afectada, multiplicador_severidad, 1.0)
        severidad = severidad * multiplicador[None, :]

        gasto_simulado = ocurrencia * severidad
        perdidas_agregadas[inicio:fin] = gasto_simulado.sum(axis=1)

    print(f"[stress_test] Pérdida agregada — media: ${perdidas_agregadas.mean():,.0f}")
    print(f"[stress_test] Pérdida agregada — desviación estándar: ${perdidas_agregadas.std():,.0f}")

    return perdidas_agregadas


# =========================================================
# 3. COMPARACIÓN BASELINE vs ESCENARIOS DE ESTRÉS
# =========================================================
def comparar_escenarios(perdidas_baseline, escenarios_estres, niveles_confianza=(0.95, 0.99)):
    """
    escenarios_estres: dict {nombre_escenario: array_perdidas}
    Genera tabla comparativa de VaR, CVaR y capital económico entre
    baseline y cada escenario de estrés.
    """
    filas = []

    tabla_base, perdida_esperada_base = calcular_var_cvar(perdidas_baseline, niveles_confianza)
    for _, row in tabla_base.iterrows():
        filas.append({
            "escenario": "Baseline",
            "nivel_confianza": row["nivel_confianza"],
            "perdida_esperada": round(perdida_esperada_base, 0),
            "VaR": row["VaR"],
            "CVaR": row["CVaR"],
            "capital_economico": row["capital_economico"],
        })

    for nombre, perdidas in escenarios_estres.items():
        tabla_esc, perdida_esperada_esc = calcular_var_cvar(perdidas, niveles_confianza)
        for _, row in tabla_esc.iterrows():
            filas.append({
                "escenario": nombre,
                "nivel_confianza": row["nivel_confianza"],
                "perdida_esperada": round(perdida_esperada_esc, 0),
                "VaR": row["VaR"],
                "CVaR": row["CVaR"],
                "capital_economico": row["capital_economico"],
            })

    tabla_comparativa = pd.DataFrame(filas)

    tabla_comparativa["capital_vs_baseline_pct"] = tabla_comparativa.groupby("nivel_confianza")[
        "capital_economico"
    ].transform(lambda x: ((x / x.iloc[0]) - 1) * 100).round(1)

    print("\n\nComparación de escenarios — baseline vs estrés:")
    print(tabla_comparativa.to_string(index=False))
    print("\n→ capital_vs_baseline_pct: incremento porcentual de capital económico "
          "requerido respecto al baseline, al mismo nivel de confianza")

    return tabla_comparativa


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Preparando parámetros de simulación (mismos de montecarlo.py)...")
    df_params = preparar_parametros_simulacion(df)

    print("\n>>> Simulando baseline (independiente, sin estrés)...")
    from src.segmentation.montecarlo import simular_perdida_portafolio
    perdidas_baseline = simular_perdida_portafolio(df_params)

    print("\n>>> Simulando escenario PANDEMIA (choque sistémico correlacionado)...")
    perdidas_pandemia = simular_escenario_pandemia(df_params)

    print("\n>>> Simulando escenario CATÁSTROFE REGIONAL...")
    perdidas_catastrofe = simular_escenario_catastrofe_regional(df_params)

    print("\n>>> Comparando escenarios...")
    tabla_comparativa = comparar_escenarios(
        perdidas_baseline,
        {"Pandemia (rho=0.15)": perdidas_pandemia, "Catástrofe regional": perdidas_catastrofe}
    )