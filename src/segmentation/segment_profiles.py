"""
segment_profiles.py
----------------------
Perfiles de segmento — caracterización y coste esperado por cluster.
Tercer módulo de la Parte 3 (Risk segmentation).

Corrección de alineación de índices (detectada tras %mujeres idéntico
en ambos clusters — señal de bug): construir_matriz_clustering() ya
NO resetea el índice tras dropna(), preservando la correspondencia
real con las filas de df al reincorporar variables de perfil y costo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]

K_SEGMENTOS = 2

FEATURES_CLUSTERING_NUMERICAS = [
    "edad", "salud_general", "salud_mental", "n_diagnosticos_unicos",
]
FEATURES_CLUSTERING_CATEGORICAS = [
    "tipo_cobertura", "categoria_pobreza",
]
FEATURES_CLUSTERING_BINARIAS = [
    "sin_seguro_anual",
    "dx_hipertension", "dx_diabetes", "dx_asma", "dx_cancer",
    "dx_artritis", "dx_cardiopatia", "dx_ictus", "dx_enfisema",
]
FEATURES_CLUSTERING = (
    FEATURES_CLUSTERING_NUMERICAS + FEATURES_CLUSTERING_CATEGORICAS + FEATURES_CLUSTERING_BINARIAS
)

VARS_PERFIL_ADICIONALES = [
    "sexo_femenino", "raza", "region", "estado_civil", "años_educacion", "nacido_usa",
]


# =========================================================
# 1. MATRIZ DE CLUSTERING — CORRIGE ALINEACIÓN DE ÍNDICE
# =========================================================
def construir_matriz_clustering(df):
    """
    IMPORTANTE: preserva el índice original de df (no resetea tras
    dropna) para poder reincorporar correctamente variables de perfil
    y costo más adelante, sin desalineación.
    """
    df_model = df[FEATURES_CLUSTERING + ["persona_id"]].dropna()  # índice original preservado

    scaler = StandardScaler()
    X_numericas = scaler.fit_transform(df_model[FEATURES_CLUSTERING_NUMERICAS])
    X_numericas = pd.DataFrame(X_numericas, columns=FEATURES_CLUSTERING_NUMERICAS, index=df_model.index)

    X_categoricas = pd.get_dummies(
        df_model[FEATURES_CLUSTERING_CATEGORICAS].astype(int).astype(str),
        prefix=FEATURES_CLUSTERING_CATEGORICAS
    ).astype(float)
    X_categoricas.index = df_model.index

    df_binarias = df_model[FEATURES_CLUSTERING_BINARIAS].copy()
    df_binarias["sin_seguro_anual"] = (df_binarias["sin_seguro_anual"] == 1).astype(float)
    X_binarias = df_binarias.astype(float)
    X_binarias.index = df_model.index

    X = pd.concat([X_numericas, X_categoricas, X_binarias], axis=1)

    print(f"[clustering] Matriz construida: {X.shape[0]:,} personas, {X.shape[1]} columnas")

    return X, df_model


def entrenar_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    print(f"\n[K-Means] Entrenado con k={k}")
    print(f"[K-Means] Distribución de clusters:")
    print(pd.Series(labels).value_counts().sort_index().to_string())
    return km, labels


# =========================================================
# 2. GENERAR CLUSTERS Y REINCORPORAR VARIABLES — SIN DESALINEACIÓN
# =========================================================
def generar_clusters(df, k=K_SEGMENTOS):
    """
    Genera los clusters y reincorpora TODAS las columnas necesarias
    (perfil demográfico + gasto/peso + FEATURES completas de
    cost_drivers, para poder calcular prima_pura después) usando el
    índice original preservado — sin riesgo de desalineación.
    """
    from src.analytics.cost_drivers import FEATURES

    X, df_model = construir_matriz_clustering(df)
    modelo_km, labels = entrenar_kmeans(X, k=k)

    df_model = df_model.copy()
    df_model["cluster"] = labels

    cols_extra = list(set(
        VARS_PERFIL_ADICIONALES + ["gasto_total_anual", "peso_muestral"] + FEATURES
    ) - set(df_model.columns))

    df_extra = df.loc[df_model.index, cols_extra]
    df_perfil = pd.concat([df_model, df_extra], axis=1).reset_index(drop=True)

    return df_perfil, modelo_km


# =========================================================
# 3. PERFIL DEMOGRÁFICO/CLÍNICO POR CLUSTER
# =========================================================
def perfil_demografico_clinico(df_perfil, peso="peso_muestral"):
    filas = []
    for cluster, g in df_perfil.groupby("cluster"):
        w = g[peso].values
        w = w / w.mean()

        fila = {"cluster": cluster, "n": len(g)}

        for var in ["edad", "salud_general", "salud_mental", "n_diagnosticos_unicos"]:
            fila[f"{var}_media"] = round(np.average(g[var], weights=w), 2)

        for dx in ["dx_hipertension", "dx_diabetes", "dx_asma", "dx_cancer",
                   "dx_artritis", "dx_cardiopatia", "dx_ictus", "dx_enfisema"]:
            fila[f"%{dx}"] = round(np.average(g[dx], weights=w) * 100, 1)

        fila["%sin_seguro"] = round(np.average((g["sin_seguro_anual"] == 1).astype(float), weights=w) * 100, 1)
        fila["%mujeres"] = round(np.average(g["sexo_femenino"], weights=w) * 100, 1)

        filas.append(fila)

    tabla = pd.DataFrame(filas)
    print("\nPerfil demográfico/clínico por cluster:")
    print(tabla.to_string(index=False))
    return tabla


# =========================================================
# 4. COSTE REAL Y ESPERADO POR CLUSTER
# =========================================================
def perfil_costo(df_perfil, peso="peso_muestral"):
    from src.analytics.cost_drivers import FEATURES

    filas = []
    for cluster, g in df_perfil.groupby("cluster"):
        w = g[peso].values
        w = w / w.mean()

        gasto_medio = np.average(g["gasto_total_anual"], weights=w)
        gasto_mediana = np.median(g["gasto_total_anual"])
        pct_gasto_cero = np.average((g["gasto_total_anual"] == 0).astype(float), weights=w) * 100

        filas.append({
            "cluster": cluster,
            "n": len(g),
            "gasto_real_medio": round(gasto_medio, 1),
            "gasto_real_mediana": round(gasto_mediana, 1),
            "%gasto_cero": round(pct_gasto_cero, 1),
        })

    tabla = pd.DataFrame(filas)

    try:
        from src.models.pricing_engine import (
            cargar_modelo_probabilidad, cargar_modelo_severidad,
            predecir_probabilidad_gasto, predecir_severidad
        )

        modelo_prob, _ = cargar_modelo_probabilidad()
        modelo_sev, _ = cargar_modelo_severidad()

        df_pred = df_perfil[FEATURES + ["cluster", peso]].dropna(subset=FEATURES).copy()
        prob_hat = predecir_probabilidad_gasto(df_pred, modelo=modelo_prob)
        sev_hat = predecir_severidad(df_pred, modelo=modelo_sev)
        df_pred["prima_pura"] = prob_hat * sev_hat

        prima_por_cluster = []
        for cluster, g in df_pred.groupby("cluster"):
            w = g[peso].values
            w = w / w.mean()
            prima_media = np.average(g["prima_pura"], weights=w)
            prima_por_cluster.append({"cluster": cluster, "prima_pura_media": round(prima_media, 1)})

        tabla_prima = pd.DataFrame(prima_por_cluster)
        tabla = tabla.merge(tabla_prima, on="cluster", how="left")

    except FileNotFoundError:
        print("[segment_profiles] Modelos de pricing_engine no encontrados — se omite prima_pura_media")

    print("\nCoste real (y predicho, si disponible) por cluster:")
    print(tabla.to_string(index=False))
    return tabla


# =========================================================
# 5. RESUMEN NARRATIVO
# =========================================================
def resumen_narrativo(tabla_demografico, tabla_costo):
    tabla = tabla_demografico.merge(tabla_costo, on=["cluster", "n"])

    cluster_alto_costo = tabla.loc[tabla["gasto_real_medio"].idxmax(), "cluster"]
    cluster_bajo_costo = tabla.loc[tabla["gasto_real_medio"].idxmin(), "cluster"]

    fila_alto = tabla[tabla["cluster"] == cluster_alto_costo].iloc[0]
    fila_bajo = tabla[tabla["cluster"] == cluster_bajo_costo].iloc[0]

    print(f"\nResumen narrativo:")
    print(f"  Cluster {cluster_alto_costo} (mayor coste, ${fila_alto['gasto_real_medio']:,.0f}/año): "
          f"edad media {fila_alto['edad_media']:.1f}, "
          f"n_diagnosticos media {fila_alto['n_diagnosticos_unicos_media']:.1f}, "
          f"{fila_alto['%dx_hipertension']:.0f}% hipertensión, "
          f"{fila_alto['%mujeres']:.0f}% mujeres, "
          f"{fila_alto['%sin_seguro']:.0f}% sin seguro")
    print(f"  Cluster {cluster_bajo_costo} (menor coste, ${fila_bajo['gasto_real_medio']:,.0f}/año): "
          f"edad media {fila_bajo['edad_media']:.1f}, "
          f"n_diagnosticos media {fila_bajo['n_diagnosticos_unicos_media']:.1f}, "
          f"{fila_bajo['%dx_hipertension']:.0f}% hipertensión, "
          f"{fila_bajo['%mujeres']:.0f}% mujeres, "
          f"{fila_bajo['%sin_seguro']:.0f}% sin seguro")

    return tabla


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Generando clusters (k=2)...")
    df_perfil, modelo_km = generar_clusters(df)

    print("\n>>> Calculando perfil demográfico/clínico...")
    tabla_demografico = perfil_demografico_clinico(df_perfil)

    print("\n>>> Calculando perfil de costo...")
    tabla_costo = perfil_costo(df_perfil)

    print("\n>>> Generando resumen narrativo...")
    tabla_final = resumen_narrativo(tabla_demografico, tabla_costo)