"""
conftest.py
-------------
Fixtures compartidas para los tests del proyecto HealthRisk360.
Usa datos reales del proyecto (no mocks) — apropiado para un pipeline
de portafolio concreto, donde el objetivo es detectar regresiones
sobre datos reales, no validar una librería genérica.
"""

import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def df_longitudinal():
    """Dataset longitudinal completo (2018-2023), cargado una sola vez."""
    path = ROOT / "data" / "processed" / "meps_2018_2023.csv"
    if not path.exists():
        pytest.skip(f"No existe {path} — correr etl.py primero")
    return pd.read_csv(path, low_memory=False)


@pytest.fixture(scope="session")
def df_2023():
    """Corte 2023, cargado una sola vez."""
    path = ROOT / "data" / "processed" / "meps_2023_clean.csv"
    if not path.exists():
        pytest.skip(f"No existe {path} — correr etl.py primero")
    return pd.read_csv(path, low_memory=False)