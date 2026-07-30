"""
3_Segmentos.py
----------------
Perfiles de segmento de riesgo (clustering K-Means, k=2) — reutiliza
segment_profiles.py directamente.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.segmentation.segment_profiles import (
    generar_clusters, perfil_demografico_clinico, perfil_costo
)

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Segmentos — HealthRisk360", page_icon="🧩", layout="wide")
st.title("🧩 Segmentos de Riesgo (Clustering)")

st.markdown("""
Segmentación K-Means (k=2) sobre variables clínicas/utilización (edad,
salud general, comorbilidades, cobertura) — sin incluir gasto ni prima
como insumo del clustering, para evitar circularidad metodológica.
El coste se calcula **después**, como propiedad emergente de cada grupo.
""")


@st.cache_data(show_spinner="Generando clusters y calculando perfiles (puede tardar ~1 min)...")
def obtener_perfiles():
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")
    df_perfil, modelo_km = generar_clusters(df)
    tabla_demografico = perfil_demografico_clinico(df_perfil)
    tabla_costo = perfil_costo(df_perfil)
    return df_perfil, tabla_demografico, tabla_costo


df_perfil, tabla_demografico, tabla_costo = obtener_perfiles()

tabla_completa = tabla_demografico.merge(tabla_costo, on=["cluster", "n"])

nombres_cluster = {
    tabla_completa.loc[tabla_completa["gasto_real_medio"].idxmin(), "cluster"]: "Bajo riesgo",
    tabla_completa.loc[tabla_completa["gasto_real_medio"].idxmax(), "cluster"]: "Alto riesgo",
}
tabla_completa["etiqueta"] = tabla_completa["cluster"].map(nombres_cluster)

st.subheader("Resumen por segmento")
cols = st.columns(len(tabla_completa))
for i, (_, fila) in enumerate(tabla_completa.iterrows()):
    with cols[i]:
        st.metric(
            f"Cluster {fila['cluster']} — {fila['etiqueta']}",
            f"${fila['gasto_real_medio']:,.0f}/año",
            f"{fila['n']:,} personas",
        )

st.subheader("Distribución de personas por cluster")

tabla_mostrar = tabla_completa.set_index("etiqueta")[["n"]]
st.bar_chart(tabla_mostrar)

col_a, col_b = st.columns(2)
total = tabla_completa["n"].sum()
for i, (_, fila) in enumerate(tabla_completa.iterrows()):
    col = col_a if i == 0 else col_b
    col.metric(f"{fila['etiqueta']}", f"{fila['n']:,} personas", f"{fila['n']/total*100:.1f}% del total")



st.subheader("Comparación de perfiles")
variables_comparar = st.multiselect(
    "Variables a comparar",
    ["edad_media", "n_diagnosticos_unicos_media", "%dx_hipertension", "%dx_diabetes",
     "%dx_cancer", "%sin_seguro", "gasto_real_medio"],
    default=["edad_media", "n_diagnosticos_unicos_media", "gasto_real_medio"],
)

if variables_comparar:
    fig_bar = go.Figure()
    for _, fila in tabla_completa.iterrows():
        fig_bar.add_trace(go.Bar(
            name=f"Cluster {fila['cluster']} ({fila['etiqueta']})",
            x=variables_comparar,
            y=[fila[v] for v in variables_comparar],
        ))
    fig_bar.update_layout(barmode="group", height=450, title="Comparación de variables por cluster")
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Perfil demográfico/clínico completo")
st.dataframe(tabla_demografico, use_container_width=True)

st.subheader("Coste real y prima del modelo por cluster")
st.dataframe(tabla_costo, use_container_width=True)

if "prima_pura_media" in tabla_costo.columns:
    tabla_costo["ratio_modelo_real"] = (tabla_costo["prima_pura_media"] / tabla_costo["gasto_real_medio"]).round(2)
    ratio_alto = tabla_costo.loc[tabla_costo["gasto_real_medio"].idxmax(), "ratio_modelo_real"]
    if ratio_alto > 1.3:
        st.warning(
            f"⚠️ El modelo sobreestima el segmento de mayor coste por un factor de {ratio_alto:.2f}x "
            f"— consistente con la limitación de calibración documentada en Desviaciones."
        )