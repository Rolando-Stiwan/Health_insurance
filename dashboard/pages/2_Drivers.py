"""
2_Drivers.py
--------------
Drivers de gasto médico (GLM Gamma) + VIF — reutiliza cost_drivers.py
directamente.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from src.analytics.cost_drivers import entrenar_glm, calcular_vif, ranking_drivers

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Drivers — HealthRisk360", page_icon="📈", layout="wide")
st.title("📈 Drivers de Gasto Médico")

st.markdown(
    "Factores que explican el gasto médico anual, según el modelo GLM Gamma "
    "(distribución seleccionada formalmente — ver decisiones_tecnicas.md)."
)


@st.cache_data(show_spinner="Entrenando modelos y calculando drivers...")
def obtener_drivers_y_vif():
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")
    resultados = entrenar_glm(df)
    vif = calcular_vif(df)
    drivers_por_modelo = ranking_drivers(resultados)
    return drivers_por_modelo, vif, resultados


drivers_por_modelo, tabla_vif, resultados = obtener_drivers_y_vif()

col1, col2 = st.columns(2)
col1.metric("RMSE Gamma (fuera de muestra)", f"${resultados['Gamma']['rmse']:,.0f}")
col2.metric("RMSE Lognormal (fuera de muestra)", f"${resultados['Lognormal']['rmse']:,.0f}")

st.subheader("Ranking de drivers — GLM Gamma")

drivers_gamma = drivers_por_modelo["Gamma"].copy()
drivers_gamma["direccion"] = drivers_gamma["exp_coef"].apply(lambda x: "Sube el gasto" if x > 1 else "Baja el gasto")
drivers_gamma_ordenado = drivers_gamma.sort_values("exp_coef")

st.markdown("""
**Cómo leer este gráfico:** el modelo predice el gasto en escala logarítmica,
así que cada coeficiente se convierte a un **multiplicador** sobre el gasto
esperado (`exp(coeficiente)`). Un valor de 1.47 significa que esa variable
multiplica el gasto por 1.47 (+47%); un valor de 0.80 significa que lo
reduce a 0.80 veces el original (-20%). La línea vertical en 1.0 marca "sin efecto".
""")

fig = go.Figure()
fig.add_trace(go.Bar(
    y=drivers_gamma_ordenado["variable"],
    x=drivers_gamma_ordenado["exp_coef"] - 1,
    base=1,
    orientation="h",
    marker_color=["crimson" if sig else "lightgray" for sig in drivers_gamma_ordenado["significativo"]],
))
fig.add_vline(x=1, line_dash="dash", line_color="gray")
fig.update_layout(
    title="Efecto multiplicador sobre el gasto esperado, por variable<br><sup>(exp(coeficiente) — 1.20 = +20% de gasto, 0.80 = -20% de gasto | rojo = significativo, p&lt;0.05)</sup>",
    xaxis_title="exp(coeficiente)",
    height=700,
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    drivers_gamma[["variable", "coeficiente", "exp_coef", "p_valor", "significativo", "direccion"]],
    use_container_width=True,
)

st.subheader("Variance Inflation Factor (VIF)")
st.markdown("VIF > 5 = revisar multicolinealidad. VIF > 10 = severo.")
st.dataframe(tabla_vif, use_container_width=True)

st.subheader("Comparación de modelos con Lognormal")
with st.expander("Ver ranking completo de Lognormal"):
    st.dataframe(drivers_por_modelo["Lognormal"], use_container_width=True)