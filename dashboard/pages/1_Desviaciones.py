"""
1_Desviaciones.py
--------------------
Desviación real/esperado (A/E) por segmento — reutiliza
deviation_analysis.py directamente (no vía API, es analítica interna).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

#from src.analytics.cost_drivers import FEATURES, entrenar_glm
from src.analytics.deviation_analysis import generar_predicciones, desviacion_por_segmento

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Desviaciones — HealthRisk360", page_icon="📊", layout="wide")
st.title("📊 Desviación Real/Esperado por Segmento")

st.markdown(
    "Compara el gasto real observado contra el gasto esperado por el modelo "
    "Gamma, agregado por segmento. Ratios alejados de 1.0 indican mala "
    "calibración del modelo en ese grupo."
)


@st.cache_data(show_spinner="Entrenando modelo y generando predicciones...")
def obtener_predicciones():
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")
    df_pred = generar_predicciones(df)
    return df_pred


df_pred = obtener_predicciones()

segmento = st.selectbox(
    "Segmento a analizar",
    ["region", "tipo_cobertura", "categoria_pobreza", "salud_general", "sin_seguro_anual"],
)

n_bootstrap = st.slider("Simulaciones bootstrap (IC 95%)", 50, 500, 200, step=50)

with st.spinner("Calculando intervalos de confianza..."):
    tabla = desviacion_por_segmento(df_pred, segmento, n_bootstrap=n_bootstrap)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=tabla["categoria"].astype(str),
    y=tabla["ratio_AE"],
    error_y=dict(
        type="data",
        symmetric=False,
        array=tabla["ic_95_high"] - tabla["ratio_AE"],
        arrayminus=tabla["ratio_AE"] - tabla["ic_95_low"],
    ),
    marker_color=["crimson" if sig else "steelblue" for sig in tabla["desviacion_significativa"]],
))
fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Calibración perfecta (A/E=1)")
fig.update_layout(
    title=f"Ratio A/E por {segmento} (rojo = desviación estadísticamente significativa)",
    xaxis_title=segmento,
    yaxis_title="Ratio Real/Esperado",
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(tabla, use_container_width=True)

n_sig = tabla["desviacion_significativa"].sum()
if n_sig > 0:
    st.warning(f"⚠️ {n_sig} categoría(s) con desviación estadísticamente significativa (IC 95% no incluye 1.0)")
else:
    st.success("✅ Ninguna categoría con desviación significativa")