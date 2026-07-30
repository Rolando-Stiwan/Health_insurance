"""
test_etl.py
-------------
Tests del pipeline ETL — priorizan las regresiones ya detectadas
durante el desarrollo (ver decisiones_tecnicas.md, sección de
corrección de datos).
"""

import pandas as pd


COLUMNAS_ESPERADAS = [
    "persona_id", "año", "edad", "sexo_femenino", "raza",
    "region", "categoria_pobreza", "ingreso_familiar",
    "tipo_cobertura", "sin_seguro_anual", "salud_general", "salud_mental",
    "gasto_total_anual", "peso_muestral",
    "dx_hipertension", "dx_diabetes", "dx_asma", "dx_cancer",
    "dx_artritis", "dx_cardiopatia", "dx_ictus", "dx_enfisema",
    "n_diagnosticos_unicos", "accidentes_trabajo",
]

CONTEOS_QUE_NO_DEBEN_TENER_NAN = [
    "n_diagnosticos_unicos", "n_sistemas_afectados",
    "lesiones", "accidentes_trabajo",
]


def test_columnas_esperadas_existen(df_longitudinal):
    """Verifica que las columnas clave del pipeline existan tras el ETL."""
    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df_longitudinal.columns]
    assert not faltantes, f"Columnas esperadas ausentes del ETL: {faltantes}"


def test_conteos_no_tienen_nan(df_longitudinal):
    """
    Regresión directa del bug detectado durante el desarrollo:
    n_diagnosticos_unicos (y otros conteos) tenían NaN espurios en
    meps_2023_clean.csv por un archivo derivado desactualizado.
    etl.py aplica fillna(0) explícitamente — este test lo confirma.
    """
    for col in CONTEOS_QUE_NO_DEBEN_TENER_NAN:
        if col in df_longitudinal.columns:
            n_nan = df_longitudinal[col].isna().sum()
            assert n_nan == 0, f"{col} tiene {n_nan} valores NaN — el fillna(0) no se aplicó correctamente"


def test_anios_completos(df_longitudinal):
    """El longitudinal debe cubrir los 6 años esperados, 2018-2023."""
    anios_esperados = {2018, 2019, 2020, 2021, 2022, 2023}
    anios_presentes = set(df_longitudinal["año"].unique())
    assert anios_esperados == anios_presentes, (
        f"Años esperados {anios_esperados}, encontrados {anios_presentes}"
    )


def test_gasto_no_negativo(df_longitudinal):
    """El gasto médico anual nunca debería ser negativo."""
    assert (df_longitudinal["gasto_total_anual"] >= 0).all(), (
        "Se encontraron valores negativos en gasto_total_anual"
    )


def test_flags_diagnostico_binarios(df_longitudinal):
    """
    Los flags dx_* deben ser 0, 1, o NaN (por códigos reservados de
    MEPS en población pediátrica) — nunca otro valor.
    """
    flags_dx = [
        "dx_hipertension", "dx_diabetes", "dx_asma", "dx_cancer",
        "dx_artritis", "dx_cardiopatia", "dx_ictus", "dx_enfisema",
    ]
    for col in flags_dx:
        valores_unicos = set(df_longitudinal[col].dropna().unique())
        assert valores_unicos <= {0, 1}, (
            f"{col} tiene valores fuera de {{0, 1}}: {valores_unicos}"
        )


def test_corte_2023_es_subconjunto_del_longitudinal(df_longitudinal, df_2023):
    """
    Regresión del bug de meps_2023_clean.csv desactualizado: confirma
    que el corte 2023 se deriva correctamente del longitudinal (mismo
    número de filas que año==2023 en el longitudinal).
    """
    n_2023_en_longitudinal = (df_longitudinal["año"] == 2023).sum()
    assert len(df_2023) == n_2023_en_longitudinal, (
        f"meps_2023_clean.csv tiene {len(df_2023)} filas, pero el longitudinal "
        f"tiene {n_2023_en_longitudinal} filas para año=2023 — posible "
        f"desincronización entre ambos archivos"
    )


def test_no_duplicados_persona_anio(df_longitudinal):
    """Una misma persona no debería aparecer dos veces en el mismo año."""
    duplicados = df_longitudinal.duplicated(subset=["persona_id", "año"]).sum()
    assert duplicados == 0, f"Se encontraron {duplicados} filas duplicadas de persona_id+año"