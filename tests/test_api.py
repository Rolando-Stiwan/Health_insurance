"""
test_api.py
-------------
Tests de los endpoints de la API FastAPI (api/main.py). Usa TestClient
de FastAPI, que dispara el lifespan (carga de modelos) automáticamente
al usarse como context manager — sin esto, los modelos nunca se
cargarían y todos los endpoints fallarían.

Requiere que los modelos ya estén entrenados y persistidos en
models_artifacts/ (severity_gamma, probabilidad_gasto_logit,
risk_lightgbm) — se skippean automáticamente si no existen.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "models_artifacts"


def _modelos_disponibles():
    archivos_esperados = [
        "severity_gamma.joblib",
        "probabilidad_gasto_logit.joblib",
        "risk_lightgbm.joblib",
    ]
    return all((ARTIFACTS_DIR / f).exists() for f in archivos_esperados)


pytestmark = pytest.mark.skipif(
    not _modelos_disponibles(),
    reason="Modelos no entrenados aún — correr claim_sev.py, pricing_engine.py "
           "y risk_detection.py primero",
)


PERSONA_VALIDA = {
    "edad": 45, "sexo_femenino": 1, "raza": 1, "estado_civil": 1,
    "region": 3, "años_educacion": 12, "nacido_usa": 1,
    "categoria_pobreza": 4, "ingreso_familiar": 45000,
    "tipo_cobertura": 1, "sin_seguro_anual": 2,
    "salud_general": 2, "salud_mental": 2,
    "dx_hipertension": 1, "dx_diabetes": 0, "dx_asma": 0,
    "dx_cancer": 0, "dx_artritis": 0, "dx_cardiopatia": 0,
    "dx_ictus": 0, "dx_enfisema": 0,
    "n_diagnosticos_unicos": 2, "accidentes_trabajo": 0,
}


@pytest.fixture(scope="module")
def client():
    """
    TestClient como context manager — dispara lifespan (startup:
    carga de modelos; shutdown: limpieza) automáticamente.
    scope="module" para cargar los modelos una sola vez para todos
    los tests de este archivo, no en cada test individual.
    """
    from api.main import app
    with TestClient(app) as c:
        yield c


# =========================================================
# /health
# =========================================================
def test_health_devuelve_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_confirma_modelos_cargados(client):
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "ok"
    esperados = {"prob_gasto", "severidad", "riesgo", "df_referencia", "tabla_credibilidad"}
    assert esperados.issubset(set(data["modelos_cargados"]))


# =========================================================
# /riesgo
# =========================================================
def test_riesgo_devuelve_200_con_persona_valida(client):
    resp = client.post("/riesgo", json=PERSONA_VALIDA)
    assert resp.status_code == 200


def test_riesgo_estructura_respuesta(client):
    resp = client.post("/riesgo", json=PERSONA_VALIDA)
    data = resp.json()
    assert "probabilidad_alto_costo" in data
    assert "clasificacion" in data
    assert 0 <= data["probabilidad_alto_costo"] <= 1
    assert data["clasificacion"] in ["Alto riesgo", "Riesgo estándar"]


def test_riesgo_falla_con_campo_faltante(client):
    persona_incompleta = {k: v for k, v in PERSONA_VALIDA.items() if k != "edad"}
    resp = client.post("/riesgo", json=persona_incompleta)
    assert resp.status_code == 422  # error de validación de Pydantic


def test_riesgo_falla_con_edad_fuera_de_rango(client):
    persona_invalida = {**PERSONA_VALIDA, "edad": 200}
    resp = client.post("/riesgo", json=persona_invalida)
    assert resp.status_code == 422  # viola Field(le=120)


# =========================================================
# /severidad
# =========================================================
def test_severidad_devuelve_200(client):
    resp = client.post("/severidad", json=PERSONA_VALIDA)
    assert resp.status_code == 200


def test_severidad_es_positiva(client):
    resp = client.post("/severidad", json=PERSONA_VALIDA)
    data = resp.json()
    assert data["severidad_esperada"] > 0


# =========================================================
# /pricing
# =========================================================
def test_pricing_devuelve_200(client):
    resp = client.post("/pricing", json=PERSONA_VALIDA)
    assert resp.status_code == 200


def test_pricing_estructura_y_coherencia(client):
    """
    Verifica estructura completa y la misma coherencia matemática que
    ya validamos en test_models.py — prima_pura no debe exceder
    severidad (prob_gasto <= 1), regresión del bug de doble conteo.
    """
    resp = client.post("/pricing", json=PERSONA_VALIDA)
    data = resp.json()

    campos_esperados = {
        "prob_gasto", "severidad_esperada", "prima_pura_individual",
        "prima_ajustada_por_segmento", "Z_credibilidad_segmento", "nota",
    }
    assert campos_esperados.issubset(set(data.keys()))

    assert 0 <= data["prob_gasto"] <= 1
    assert data["severidad_esperada"] > 0
    assert data["prima_pura_individual"] <= data["severidad_esperada"] + 0.01  # tolerancia de redondeo
    assert data["prima_pura_individual"] < 500_000, (
        "Prima pura sospechosamente alta — posible regresión del bug de doble conteo"
    )


def test_pricing_credibilidad_coherente_por_tipo_cobertura(client):
    """
    tipo_cobertura=2 (público) debería tener mayor prima ajustada que
    tipo_cobertura=1 (privado), consistente con el hallazgo ya
    documentado en deviation_analysis.py y pricing_engine.py.
    """
    persona_privado = {**PERSONA_VALIDA, "tipo_cobertura": 1}
    persona_publico = {**PERSONA_VALIDA, "tipo_cobertura": 2}

    resp_privado = client.post("/pricing", json=persona_privado).json()
    resp_publico = client.post("/pricing", json=persona_publico).json()

    assert resp_publico["prima_ajustada_por_segmento"] > resp_privado["prima_ajustada_por_segmento"]


# =========================================================
# /stress-test
# =========================================================
def test_stress_test_pandemia_devuelve_200(client):
    payload = {"escenario": "pandemia", "n_simulaciones": 200}
    resp = client.post("/stress-test", json=payload)
    assert resp.status_code == 200


def test_stress_test_catastrofe_devuelve_200(client):
    payload = {"escenario": "catastrofe_regional", "n_simulaciones": 200, "region_afectada": 3}
    resp = client.post("/stress-test", json=payload)
    assert resp.status_code == 200


def test_stress_test_estructura_respuesta(client):
    payload = {"escenario": "pandemia", "n_simulaciones": 200}
    resp = client.post("/stress-test", json=payload)
    data = resp.json()

    assert data["escenario"] == "pandemia"
    assert data["perdida_esperada"] > 0
    assert len(data["resultados"]) == 2  # niveles 95% y 99%

    for fila in data["resultados"]:
        assert fila["VaR"] > 0
        assert fila["CVaR"] >= fila["VaR"]  # CVaR siempre >= VaR por definición
        assert fila["capital_economico"] >= 0


def test_stress_test_escenario_invalido_devuelve_400(client):
    payload = {"escenario": "zombie_apocalypse", "n_simulaciones": 200}
    resp = client.post("/stress-test", json=payload)
    assert resp.status_code == 400


def test_stress_test_pandemia_mas_riesgosa_que_catastrofe(client):
    """
    Regresión del hallazgo ya documentado: el capital económico de
    pandemia (riesgo sistémico/correlacionado) debería ser
    proporcionalmente mayor que el de catástrofe regional, relativo
    a su propia pérdida esperada.
    """
    payload_pandemia = {"escenario": "pandemia", "n_simulaciones": 500}
    payload_catastrofe = {"escenario": "catastrofe_regional", "n_simulaciones": 500, "region_afectada": 3}

    data_pandemia = client.post("/stress-test", json=payload_pandemia).json()
    data_catastrofe = client.post("/stress-test", json=payload_catastrofe).json()

    capital_pct_pandemia = (
        data_pandemia["resultados"][1]["capital_economico"] / data_pandemia["perdida_esperada"]
    )
    capital_pct_catastrofe = (
        data_catastrofe["resultados"][1]["capital_economico"] / data_catastrofe["perdida_esperada"]
    )

    assert capital_pct_pandemia > capital_pct_catastrofe, (
        "Se esperaba que el riesgo sistémico (pandemia) requiriera proporcionalmente "
        "más capital económico que el riesgo localizado (catástrofe regional)"
    )