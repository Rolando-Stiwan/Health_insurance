"""
5_Stress_Test.py
-------------------
Pruebas de estrés — pandemia (choque sistémico correlacionado) y
catástrofe regional (choque localizado). Conectado en vivo a la API
FastAPI (/stress-test).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Stress Test — HealthRisk360", page_icon="🌪️", layout="wide")
st.title("🌪️ Pruebas de Estrés")

st.markdown("""
Simulación Monte Carlo de VaR, CVaR y capital económico bajo escenarios
de estrés. **Pandemia**: choque sistémico correlacionado (modelo de un
factor, mismo principio que Vasicek para riesgo de crédito). 
**Catástrofe regional**: choque de severidad localizado en una región.
""")

st.caption(
    "⚠️ Por límites de memoria del hosting gratuito, esta demo simula un "
    "portafolio de 2,000 personas (en vez del dataset completo de ~15,000) "
    "La metodología es idéntica a la documentada en "
    "decisiones_tecnicas.md — solo cambia la escala para "
    "esta demostración web."
)


try:
    resp_health = requests.get(f"{API_URL}/health", timeout=3)
    api_disponible = resp_health.status_code == 200
except requests.exceptions.ConnectionError:
    api_disponible = False

if not api_disponible:
    st.error(
        "⚠️ No se pudo conectar con la API en " + API_URL + ". "
        "Asegúrate de que esté corriendo: `uvicorn api.main:app --reload`"
    )
    st.stop()

st.success("✅ API conectada")

col_config1, col_config2, col_config3 = st.columns(3)

with col_config1:
    escenario = st.selectbox(
        "Escenario de estrés",
        options=["pandemia", "catastrofe_regional"],
        format_func=lambda x: "🦠 Pandemia" if x == "pandemia" else "🌪️ Catástrofe regional",
    )

with col_config2:
    n_simulaciones = st.select_slider(
        "Número de simulaciones",
        options=[500, 1000, 2000, 5000, 10000],
        value=2000,
        help="Más simulaciones = más precisión, pero más tiempo de espera",
    )

with col_config3:
    region_afectada = None
    if escenario == "catastrofe_regional":
        region_afectada = st.selectbox(
            "Región afectada",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Noreste", 2: "Medio Oeste", 3: "Sur", 4: "Oeste"}[x],
            index=2,
        )

if st.button("🎲 Ejecutar simulación", type="primary"):
    payload = {"escenario": escenario, "n_simulaciones": n_simulaciones}
    if region_afectada is not None:
        payload["region_afectada"] = region_afectada

    with st.spinner(f"Ejecutando {n_simulaciones:,} simulaciones... puede tardar unos segundos"):
        try:
            resp = requests.post(f"{API_URL}/stress-test", json=payload, timeout=120)
            resp.raise_for_status()
            resultado = resp.json()

            st.subheader(f"Resultado — {resultado['escenario']}")

            tabla = pd.DataFrame(resultado["resultados"])

            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Pérdida esperada del portafolio", f"${resultado['perdida_esperada']:,.0f}")

            fila_99 = tabla[tabla["nivel_confianza"] == "99%"].iloc[0]
            col_r2.metric(
                "Capital económico (99%)",
                f"${fila_99['capital_economico']:,.0f}",
                help="Buffer necesario sobre la pérdida esperada al nivel de confianza 99%",
            )

            st.dataframe(tabla, use_container_width=True)

            tabla_grafico = tabla.set_index("nivel_confianza")

            col_var, col_cvar = st.columns(2)
            with col_var:
                st.markdown("**VaR por nivel de confianza**")
                st.bar_chart(tabla_grafico[["VaR"]])
            with col_cvar:
                st.markdown("**CVaR por nivel de confianza**")
                st.bar_chart(tabla_grafico[["CVaR"]])

            st.caption(f"📌 Pérdida esperada del portafolio: ${resultado['perdida_esperada']:,.0f}")

            st.session_state.setdefault("historial_stress_test", [])
            st.session_state["historial_stress_test"].append({
                "escenario": resultado["escenario"],
                "n_simulaciones": n_simulaciones,
                "perdida_esperada": resultado["perdida_esperada"],
                "capital_99": fila_99["capital_economico"],
            })

        except requests.exceptions.RequestException as e:
            st.error(f"Error al consultar la API: {e}")

if "historial_stress_test" in st.session_state and len(st.session_state["historial_stress_test"]) > 0:
    st.subheader("Historial de esta sesión")
    st.dataframe(pd.DataFrame(st.session_state["historial_stress_test"]), use_container_width=True)
    st.caption(
        "Comparación de escenarios corridos en esta sesión. Nota: el escenario "
        "'Baseline' (independiente, sin correlación) no está disponible como "
        "endpoint separado — ver montecarlo.py para esa referencia (capital "
        "económico 99%: $64.4M)."
    )