"""
app.py
------------------
Home del dashboard HealthRisk360 — resumen del proyecto y navegación.
Primer archivo de la Parte 5 (Deployment).
"""

import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="HealthRisk360",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 HealthRisk360")
st.markdown("**Pricing de seguro médico — MEPS 2018-2023**")

st.markdown("""
Portafolio de analítica actuarial de siniestralidad en Salud, construido
sobre datos longitudinales de MEPS (Medical Expenditure Panel Survey).

### Navegación
Usa el menú lateral para explorar:
- **📊 Desviaciones** — real vs esperado por segmento (calibración del modelo)
- **📈 Drivers** — qué factores explican el gasto médico
- **🧩 Segmentos** — perfiles de riesgo (clustering)
- **💰 Pricing** — cotizador individual (conectado a la API)
- **🌪️ Stress Test** — escenarios de pandemia y catástrofe regional
""")


@st.cache_data
def cargar_resumen():
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")
    return {
        "n_personas": len(df),
        "gasto_medio": df["gasto_total_anual"].mean(),
        "pct_gasto_cero": (df["gasto_total_anual"] == 0).mean() * 100,
    }


resumen = cargar_resumen()

col1, col2, col3 = st.columns(3)
col1.metric("Personas en el dataset", f"{resumen['n_personas']:,}")
col2.metric("Gasto medio anual", f"${resumen['gasto_medio']:,.0f}")
col3.metric("% con gasto cero", f"{resumen['pct_gasto_cero']:.1f}%")

st.info(
    "⚠️ El modelo de severidad tiene una limitación de calibración conocida "
    "en la cola de alto riesgo (sobreestimación en salud_general baja y "
    "cobertura pública) — ver página de Desviaciones y decisiones_tecnicas.md."
)