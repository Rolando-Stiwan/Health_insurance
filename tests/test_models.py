"""
test_models.py
----------------
Tests de los modelos entrenados y persistidos — verifican que las
predicciones estén en rangos válidos y que las funciones fallen de
forma controlada ante inputs incompletos, en vez de fallar
silenciosamente con resultados incorrectos.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.analytics.cost_drivers import FEATURES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "models_artifacts"


def _modelos_disponibles():
    """Chequeo rápido de que los artefactos ya fueron entrenados."""
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


@pytest.fixture(scope="module")
def muestra_features(df_2023):
    """Pequeña muestra con FEATURES completas, sin NaN."""
    return df_2023[FEATURES].dropna().head(20).reset_index(drop=True)


def test_severidad_siempre_positiva(muestra_features):
    """La severidad esperada (gasto dado que hay gasto) debe ser > 0."""
    from src.models.claim_sev import predecir_severidad
    pred = predecir_severidad(muestra_features)
    assert (pred > 0).all(), "Se encontraron predicciones de severidad <= 0"


def test_severidad_en_rango_razonable(muestra_features):
    """
    Chequeo de sanidad: la severidad no debería predecir valores
    absurdos (ej. billones de dólares) para un individuo.
    """
    from src.models.claim_sev import predecir_severidad
    pred = predecir_severidad(muestra_features)
    assert pred.max() < 1_000_000, (
        f"Predicción de severidad sospechosamente alta: ${pred.max():,.0f}"
    )


def test_probabilidad_gasto_entre_0_y_1(muestra_features):
    """P(gasto>0) debe ser una probabilidad válida."""
    from src.models.pricing_engine import predecir_probabilidad_gasto
    pred = predecir_probabilidad_gasto(muestra_features)
    assert (pred >= 0).all() and (pred <= 1).all(), (
        "Se encontraron probabilidades fuera de [0, 1]"
    )


def test_severidad_falla_con_feature_faltante(muestra_features):
    """
    predecir_severidad debe fallar explícitamente (ValueError) si falta
    una feature requerida, en vez de predecir silenciosamente con datos
    incompletos.
    """
    from src.models.claim_sev import predecir_severidad
    df_incompleto = muestra_features.drop(columns=["edad"])
    with pytest.raises(ValueError):
        predecir_severidad(df_incompleto)


def test_prima_pura_es_producto_coherente(muestra_features):
    """
    Chequeo de coherencia: prima_pura = prob_gasto x severidad, sin
    doble conteo (regresión del bug de ~120x detectado durante el
    desarrollo, cuando se combinó frecuencia de eventos con severidad
    total anual por error).
    """
    from src.models.claim_sev import predecir_severidad
    from src.models.pricing_engine import predecir_probabilidad_gasto

    prob = predecir_probabilidad_gasto(muestra_features)
    sev = predecir_severidad(muestra_features)
    prima_pura = prob * sev

    # La prima pura individual no debería superar la severidad esperada
    # (ya que prob <= 1), y debe ser razonable frente al gasto típico
    # observado en el dataset (órdenes de miles, no millones).
    assert (prima_pura <= sev).all(), (
        "prima_pura no puede ser mayor que la severidad esperada (prob<=1)"
    )
    assert prima_pura.max() < 500_000, (
        f"Prima pura sospechosamente alta: ${prima_pura.max():,.0f} — "
        f"posible reintroducción del bug de doble conteo"
    )