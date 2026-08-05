"""
6_Explainability.py
----------------------
SHAP global y local — LightGBM (riesgo) y GLM Gamma (severidad).
Reutiliza explainability.py directamente.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from src.segmentation.explainability import (
    shap_lightgbm, shap_global_lightgbm, shap_local_lightgbm,
    shap_severidad_gamma, shap_global_severidad, shap_local_severidad,
)

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Explainability — HealthRisk360", page_icon="🔍", layout="wide")
st.title("🔍 Explicabilidad (SHAP)")

st.markdown("""
Qué variables explican cada predicción, según SHAP (SHapley Additive
exPlanations). **Global**: importancia promedio sobre muchas personas.
**Local**: desglose de una predicción individual específica.
""")

modelo_elegido = st.radio(
    "Modelo a explicar",
    options=["Alto riesgo (LightGBM)", "Severidad (GLM Gamma)"],
    horizontal=True,
)


@st.cache_data(show_spinner="Calculando SHAP para LightGBM (rápido)...")
def calcular_shap_lightgbm():
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")
    _, shap_values, X_muestra = shap_lightgbm(df, n_muestra=1000)
    importancia = shap_global_lightgbm(shap_values, X_muestra)
    return shap_values, X_muestra, importancia


@st.cache_data(show_spinner="Calculando SHAP para GLM Gamma (puede tardar ~10-30 seg)...")
def calcular_shap_severidad():
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")
    _, shap_values, X_evaluar = shap_severidad_gamma(df, n_muestra=150, n_background=80)
    importancia = shap_global_severidad(shap_values, X_evaluar)
    return shap_values, X_evaluar, importancia


if modelo_elegido == "Alto riesgo (LightGBM)":
    shap_values, X_muestra, importancia = calcular_shap_lightgbm()
    unidad = "impacto en probabilidad de alto riesgo"
else:
    shap_values, X_muestra, importancia = calcular_shap_severidad()
    unidad = "impacto en gasto esperado ($)"

st.subheader(f"SHAP Global — {modelo_elegido}")
st.caption(f"Importancia promedio de cada variable — {unidad}")

top_n = st.slider("Número de variables a mostrar", 5, 20, 10)
importancia_top = importancia.head(top_n).sort_values("shap_importancia_media")

fig_global = go.Figure()
fig_global.add_trace(go.Bar(
    x=importancia_top["shap_importancia_media"],
    y=importancia_top["variable"],
    orientation="h",
    marker_color="#4C78A8",
))
fig_global.update_layout(
    height=max(350, top_n * 30),
    xaxis_title="Importancia SHAP media (|valor absoluto|)",
    margin=dict(l=10, r=10, t=20, b=20),
)
st.plotly_chart(fig_global, use_container_width=True)

st.dataframe(importancia, use_container_width=True)

st.markdown("---")
st.subheader(f"SHAP Local — {modelo_elegido}")
st.caption("Explicación de la predicción para una persona específica de la muestra.")

idx_persona = st.slider("Índice de persona en la muestra evaluada", 0, len(X_muestra) - 1, 0)

fila_shap = shap_values[idx_persona]
fila_valores = X_muestra.iloc[idx_persona]

tabla_local = pd.DataFrame({
    "variable": X_muestra.columns,
    "valor_persona": fila_valores.values,
    "shap_value": fila_shap,
})
tabla_local["abs_shap"] = tabla_local["shap_value"].abs()
tabla_local = tabla_local.sort_values("abs_shap", ascending=False).head(10)

fig_local = go.Figure()
fig_local.add_trace(go.Bar(
    x=tabla_local["shap_value"],
    y=tabla_local["variable"],
    orientation="h",
    marker_color=["crimson" if v > 0 else "steelblue" for v in tabla_local["shap_value"]],
    text=tabla_local["valor_persona"].apply(lambda x: f"valor={x}"),
    textposition="outside",
))
fig_local.add_vline(x=0, line_color="gray")
fig_local.update_layout(
    height=400,
    xaxis_title=f"SHAP value ({unidad})",
    margin=dict(l=10, r=80, t=20, b=20),
)
st.plotly_chart(fig_local, use_container_width=True)

st.caption(
    "🔴 Rojo: la variable empuja la predicción hacia arriba (más riesgo/gasto). "
    "🔵 Azul: la empuja hacia abajo."
)

st.dataframe(tabla_local[["variable", "valor_persona", "shap_value"]], use_container_width=True)