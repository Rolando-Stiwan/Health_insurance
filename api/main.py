"""
main.py
------------
API REST con FastAPI — expone los modelos ya entrenados y persistidos
del proyecto HealthRisk360.

Endpoints:
- GET  /health              — chequeo de estado
- POST /riesgo               — probabilidad de alto costo (LightGBM)
- POST /severidad            — gasto esperado dado que hay gasto (Gamma)
- POST /pricing               — prima pura (P(gasto>0) x severidad) +
                                 prima ajustada por credibilidad de segmento
- POST /stress-test           — VaR/CVaR bajo escenario pandemia o
                                 catástrofe regional, sobre una muestra
                                 del dataset (no requiere input individual)

Todos los modelos se cargan UNA VEZ al iniciar la API (startup), no en
cada request — consistente con la arquitectura training/inference ya
establecida en claim_sev.py, claim_freq.py y pricing_engine.py.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from pathlib import Path

from src.analytics.cost_drivers import FEATURES
from src.models.claim_sev import cargar_modelo_severidad, predecir_severidad
from src.models.pricing_engine import (
    cargar_modelo_probabilidad, predecir_probabilidad_gasto,
    calcular_credibilidad_buhlmann
)
from src.segmentation.risk_detection import cargar_modelo_lightgbm
from src.segmentation.montecarlo import preparar_parametros_simulacion, simular_perdida_portafolio, calcular_var_cvar
from src.segmentation.stress_test import simular_escenario_pandemia, simular_escenario_catastrofe_regional

ROOT = Path(__file__).resolve().parents[1]

modelos = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga todos los modelos una vez al iniciar la API."""
    print("[api] Cargando modelos...")
    modelos["prob_gasto"], _ = cargar_modelo_probabilidad()
    modelos["severidad"], _ = cargar_modelo_severidad()
    modelos["riesgo"], _ = cargar_modelo_lightgbm()

    modelos["df_referencia"] = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    df_prima = modelos["df_referencia"][FEATURES + ["persona_id", "peso_muestral", "gasto_total_anual"]].dropna(subset=FEATURES).copy()
    #df_prima = modelos["df_referencia"][FEATURES + ["persona_id", "peso_muestral", "gasto_total_anual", "tipo_cobertura"]].dropna(subset=FEATURES).copy()
    prob_hat = predecir_probabilidad_gasto(df_prima, modelo=modelos["prob_gasto"])
    sev_hat = predecir_severidad(df_prima, modelo=modelos["severidad"])
    df_prima["prima_pura"] = prob_hat * sev_hat
    tabla_cred, _ = calcular_credibilidad_buhlmann(df_prima, basado_en="modelo")
    modelos["tabla_credibilidad"] = tabla_cred

    print("[api] Modelos cargados correctamente")
    yield
    modelos.clear()


app = FastAPI(
    title="HealthRisk360 API",
    description="Pricing de seguro médico — MEPS 2018-2023",
    version="0.1.0",
    lifespan=lifespan,
)


# =========================================================
# ESQUEMA DE ENTRADA — UNA PERSONA
# =========================================================
class PersonaInput(BaseModel):
    edad: float = Field(..., ge=0, le=120)
    sexo_femenino: int = Field(..., ge=0, le=1)
    raza: float
    estado_civil: float
    region: float = Field(..., ge=1, le=4)
    años_educacion: float
    nacido_usa: float
    categoria_pobreza: float = Field(..., ge=1, le=5)
    ingreso_familiar: float
    tipo_cobertura: float = Field(..., ge=1, le=3)
    sin_seguro_anual: float = Field(..., ge=1, le=2)
    salud_general: float = Field(..., ge=1, le=5)
    salud_mental: float = Field(..., ge=1, le=5)
    dx_hipertension: float = Field(..., ge=0, le=1)
    dx_diabetes: float = Field(..., ge=0, le=1)
    dx_asma: float = Field(..., ge=0, le=1)
    dx_cancer: float = Field(..., ge=0, le=1)
    dx_artritis: float = Field(..., ge=0, le=1)
    dx_cardiopatia: float = Field(..., ge=0, le=1)
    dx_ictus: float = Field(..., ge=0, le=1)
    dx_enfisema: float = Field(..., ge=0, le=1)
    n_diagnosticos_unicos: float = Field(..., ge=0)
    accidentes_trabajo: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


def _persona_a_dataframe(persona: PersonaInput) -> pd.DataFrame:
    return pd.DataFrame([persona.model_dump()])[FEATURES]


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok", "modelos_cargados": list(modelos.keys())}


@app.post("/riesgo")
def endpoint_riesgo(persona: PersonaInput):
    """Probabilidad de alto costo (top 20% de gasto) — LightGBM."""
    try:
        df = _persona_a_dataframe(persona)
        prob = modelos["riesgo"].predict_proba(df)[:, 1][0]
        return {
            "probabilidad_alto_costo": round(float(prob), 4),
            "clasificacion": "Alto riesgo" if prob >= 0.5 else "Riesgo estándar",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/severidad")
def endpoint_severidad(persona: PersonaInput):
    """Gasto esperado dado que hay gasto — GLM Gamma."""
    try:
        df = _persona_a_dataframe(persona)
        sev = predecir_severidad(df, modelo=modelos["severidad"])
        return {"severidad_esperada": round(float(sev.iloc[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pricing")
def endpoint_pricing(persona: PersonaInput):
    """
    Prima pura individual (P(gasto>0) x severidad) y prima ajustada
    por credibilidad de Bühlmann según tipo_cobertura.
    """
    try:
        df = _persona_a_dataframe(persona)
        prob = predecir_probabilidad_gasto(df, modelo=modelos["prob_gasto"]).iloc[0]
        sev = predecir_severidad(df, modelo=modelos["severidad"]).iloc[0]
        prima_pura = prob * sev

        tabla_cred = modelos["tabla_credibilidad"]
        fila_cred = tabla_cred[tabla_cred["categoria"] == persona.tipo_cobertura]
        if len(fila_cred) == 0:
            prima_final = prima_pura
            z_credibilidad = None
        else:
            prima_final = float(fila_cred["prima_credibilidad"].iloc[0])
            z_credibilidad = float(fila_cred["Z_credibilidad"].iloc[0])

        return {
            "prob_gasto": round(float(prob), 4),
            "severidad_esperada": round(float(sev), 2),
            "prima_pura_individual": round(float(prima_pura), 2),
            "prima_ajustada_por_segmento": round(prima_final, 2),
            "Z_credibilidad_segmento": round(z_credibilidad, 4) if z_credibilidad else None,
            "nota": "prima_ajustada_por_segmento usa credibilidad de Bühlmann por "
                    "tipo_cobertura, ver decisiones_tecnicas.md",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class StressTestInput(BaseModel):
    escenario: str = Field(..., description="'pandemia' o 'catastrofe_regional'")
    n_simulaciones: int = Field(default=5000, ge=100, le=20000)
    region_afectada: float = Field(default=3, description="Solo usado si escenario='catastrofe_regional'")


@app.post("/stress-test")
def endpoint_stress_test(input_data: StressTestInput):
    """
    Corre el escenario de estrés solicitado sobre el portafolio de
    referencia (dataset completo cargado en memoria) y devuelve
    VaR/CVaR/capital económico.
    """
    try:
        df_params = preparar_parametros_simulacion(modelos["df_referencia"])

        if input_data.escenario == "pandemia":
            perdidas = simular_escenario_pandemia(df_params, n_simulaciones=input_data.n_simulaciones)
        elif input_data.escenario == "catastrofe_regional":
            perdidas = simular_escenario_catastrofe_regional(
                df_params, region_afectada=input_data.region_afectada,
                n_simulaciones=input_data.n_simulaciones
            )
        else:
            raise HTTPException(status_code=400, detail="escenario debe ser 'pandemia' o 'catastrofe_regional'")

        tabla_riesgo, perdida_esperada = calcular_var_cvar(perdidas)

        return {
            "escenario": input_data.escenario,
            "perdida_esperada": round(float(perdida_esperada), 2),
            "resultados": tabla_riesgo.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))