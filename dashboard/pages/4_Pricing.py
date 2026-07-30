"""
4_Pricing.py
--------------
Cotizador individual — formulario visual conectado a la API FastAPI
(/pricing). Demuestra integración full-stack: Streamlit como frontend,
FastAPI como backend de modelos.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pricing — HealthRisk360", page_icon="💰", layout="wide")
st.title("💰 Cotizador Individual")

st.markdown(
    "Calcula la prima pura individual (P(gasto>0) × severidad) y la prima "
    "ajustada por credibilidad de Bühlmann, según el tipo de cobertura. "
    "Conectado en vivo a la API FastAPI (`/pricing`)."
)

# Chequeo de que la API esté disponible
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

st.subheader("Datos de la persona")

col1, col2, col3 = st.columns(3)

with col1:
    edad = st.number_input("Edad", min_value=0, max_value=120, value=45)
    sexo_femenino = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Mujer" if x == 1 else "Hombre")
    raza = st.number_input("Raza (código MEPS)", min_value=1, max_value=6, value=1)
    estado_civil = st.number_input("Estado civil (código MEPS)", min_value=1, max_value=8, value=1)
    region = st.selectbox("Región", options=[1, 2, 3, 4],
                          format_func=lambda x: {1: "Noreste", 2: "Medio Oeste", 3: "Sur", 4: "Oeste"}[x])
    años_educacion = st.number_input("Años de educación", min_value=0, max_value=20, value=12)
    nacido_usa = st.selectbox("Nacido en EE.UU.", options=[1, 2], format_func=lambda x: "Sí" if x == 1 else "No")

with col2:
    categoria_pobreza = st.selectbox(
        "Categoría de pobreza", options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Pobre", 2: "Casi pobre", 3: "Ingreso bajo",
                                4: "Ingreso medio", 5: "Ingreso alto"}[x],
        index=3,
    )
    ingreso_familiar = st.number_input("Ingreso familiar anual ($)", min_value=0, value=45000, step=1000)
    tipo_cobertura = st.selectbox(
        "Tipo de cobertura", options=[1, 2, 3],
        format_func=lambda x: {1: "Privado", 2: "Público solamente", 3: "Sin seguro"}[x],
    )
    sin_seguro_anual = st.selectbox(
        "Sin seguro todo el año", options=[1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No", index=1,
    )
    salud_general = st.selectbox(
        "Salud general", options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Excelente", 2: "Muy buena", 3: "Buena",
                                4: "Regular", 5: "Mala"}[x],
        index=1,
    )
    salud_mental = st.selectbox(
        "Salud mental", options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Excelente", 2: "Muy buena", 3: "Buena",
                                4: "Regular", 5: "Mala"}[x],
        index=1,
    )

with col3:
    st.markdown("**Diagnósticos**")
    dx_hipertension = int(st.checkbox("Hipertensión", value=True))
    dx_diabetes = int(st.checkbox("Diabetes"))
    dx_asma = int(st.checkbox("Asma"))
    dx_cancer = int(st.checkbox("Cáncer"))
    dx_artritis = int(st.checkbox("Artritis"))
    dx_cardiopatia = int(st.checkbox("Cardiopatía"))
    dx_ictus = int(st.checkbox("Ictus"))
    dx_enfisema = int(st.checkbox("Enfisema"))
    accidentes_trabajo = st.number_input("Accidentes de trabajo (conteo)", min_value=0, value=0)

n_diagnosticos_unicos = sum([
    dx_hipertension, dx_diabetes, dx_asma, dx_cancer,
    dx_artritis, dx_cardiopatia, dx_ictus, dx_enfisema,
])
st.caption(f"n_diagnosticos_unicos calculado automáticamente: {n_diagnosticos_unicos}")

if st.button("💰 Calcular Prima", type="primary"):
    payload = {
        "edad": edad, "sexo_femenino": sexo_femenino, "raza": raza,
        "estado_civil": estado_civil, "region": region, "años_educacion": años_educacion,
        "nacido_usa": nacido_usa, "categoria_pobreza": categoria_pobreza,
        "ingreso_familiar": ingreso_familiar, "tipo_cobertura": tipo_cobertura,
        "sin_seguro_anual": sin_seguro_anual, "salud_general": salud_general,
        "salud_mental": salud_mental, "dx_hipertension": dx_hipertension,
        "dx_diabetes": dx_diabetes, "dx_asma": dx_asma, "dx_cancer": dx_cancer,
        "dx_artritis": dx_artritis, "dx_cardiopatia": dx_cardiopatia,
        "dx_ictus": dx_ictus, "dx_enfisema": dx_enfisema,
        "n_diagnosticos_unicos": n_diagnosticos_unicos,
        "accidentes_trabajo": accidentes_trabajo,
    }

    with st.spinner("Consultando la API..."):
        try:
            resp = requests.post(f"{API_URL}/pricing", json=payload, timeout=10)
            resp.raise_for_status()
            resultado = resp.json()

            st.subheader("Resultado")

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Probabilidad de gasto", f"{resultado['prob_gasto']*100:.1f}%")
            col_r2.metric("Severidad esperada", f"${resultado['severidad_esperada']:,.0f}")
            col_r3.metric("Prima pura individual", f"${resultado['prima_pura_individual']:,.0f}")

            st.markdown("---")
            st.metric(
                "💰 Prima ajustada por segmento (credibilidad de Bühlmann)",
                f"${resultado['prima_ajustada_por_segmento']:,.0f}",
                help=f"Z de credibilidad del segmento: {resultado['Z_credibilidad_segmento']}",
            )
            st.caption(resultado["nota"])

            if resultado["prima_ajustada_por_segmento"] > resultado["prima_pura_individual"] * 1.3:
                st.warning(
                    "⚠️ La prima de segmento es notablemente mayor que la prima pura individual — "
                    "revisar limitación de calibración documentada en Desviaciones."
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Error al consultar la API: {e}")