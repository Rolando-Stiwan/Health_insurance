"""
clustering.py
--------------
Segmentación de riesgo — K-Means (modelo final) + jerárquico (validación).
Primer módulo de la Parte 3 (Risk segmentation).

Metodología:
1. Clustering basado en variables de riesgo clínico/utilización — versión
   reducida tras detectar silhouette bajo (0.14) y ARI bajo (0.06) con el
   set completo de 51 columnas codificadas. Se eliminan variables
   demográficas de bajo valor discriminativo para riesgo (raza,
   estado_civil, region, sexo_femenino, años_educacion, accidentes_trabajo),
   que generaban muchas columnas dispersas (curse of dimensionality) sin
   aportar estructura clara.
2. No incluye prima_pura ni ninguna salida de modelo de costo — evita
   circularidad metodológica.
3. Selección de K vía método del codo (inercia) + silhouette score.
4. K-Means como modelo final. Jerárquico (Ward) sobre submuestra, solo
   para validación (ver nota de complejidad O(n²) en versión anterior).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]

# Variables de riesgo clínico/utilización — versión reducida y enfocada.
FEATURES_CLUSTERING_NUMERICAS = [
    "edad", "salud_general", "salud_mental", "n_diagnosticos_unicos",
]

FEATURES_CLUSTERING_CATEGORICAS = [
    "tipo_cobertura", "categoria_pobreza",
]

FEATURES_CLUSTERING_BINARIAS = [
    "sin_seguro_anual",  # ya es 1/2, se recodea a 0/1 en construir_matriz_clustering
    "dx_hipertension", "dx_diabetes", "dx_asma", "dx_cancer",
    "dx_artritis", "dx_cardiopatia", "dx_ictus", "dx_enfisema",
]

FEATURES_CLUSTERING = (
    FEATURES_CLUSTERING_NUMERICAS + FEATURES_CLUSTERING_CATEGORICAS + FEATURES_CLUSTERING_BINARIAS
)


# =========================================================
# 1. CONSTRUCCIÓN DE LA MATRIZ DE CLUSTERING
# =========================================================
def construir_matriz_clustering(df):
    """
    Escala variables numéricas/ordinales (StandardScaler), one-hot
    encodea tipo_cobertura y categoria_pobreza, y recodea
    sin_seguro_anual (1/2) a binario 0/1 estándar.
    """
    df_model = df[FEATURES_CLUSTERING + ["persona_id"]].dropna().reset_index(drop=True)

    scaler = StandardScaler()
    X_numericas = scaler.fit_transform(df_model[FEATURES_CLUSTERING_NUMERICAS])
    X_numericas = pd.DataFrame(X_numericas, columns=FEATURES_CLUSTERING_NUMERICAS)

    X_categoricas = pd.get_dummies(
        df_model[FEATURES_CLUSTERING_CATEGORICAS].astype(int).astype(str),
        prefix=FEATURES_CLUSTERING_CATEGORICAS
    ).astype(float)

    df_binarias = df_model[FEATURES_CLUSTERING_BINARIAS].copy()
    df_binarias["sin_seguro_anual"] = (df_binarias["sin_seguro_anual"] == 1).astype(float)
    X_binarias = df_binarias.astype(float).reset_index(drop=True)

    X = pd.concat([X_numericas, X_categoricas, X_binarias], axis=1)

    print(f"[clustering] Matriz construida: {X.shape[0]:,} personas, {X.shape[1]} columnas "
          f"(tras codificación, versión reducida)")

    return X, df_model


# =========================================================
# 2. SELECCIÓN DE K — CODO + SILHOUETTE
# =========================================================
def determinar_k_optimo(X, k_range=range(2, 11), muestra_silhouette=3000, random_state=42):
    rng = np.random.default_rng(random_state)
    resultados = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)

        inertia = km.inertia_

        n_muestra = min(muestra_silhouette, len(X))
        idx_muestra = rng.choice(len(X), size=n_muestra, replace=False)
        sil = silhouette_score(X.iloc[idx_muestra], labels[idx_muestra])

        resultados.append({"k": k, "inercia": round(inertia, 1), "silhouette": round(sil, 4)})
        print(f"  k={k} → inercia={inertia:,.1f}  silhouette={sil:.4f}")

    tabla = pd.DataFrame(resultados)
    k_optimo_silhouette = tabla.loc[tabla["silhouette"].idxmax(), "k"]

    print(f"\n→ K con mayor silhouette: {k_optimo_silhouette}")
    print("→ Revisar también el codo (caída marginal de inercia) antes de decidir K final")

    return tabla, k_optimo_silhouette


# =========================================================
# 3. ENTRENAMIENTO K-MEANS (MODELO FINAL)
# =========================================================
def entrenar_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    print(f"\n[K-Means] Entrenado con k={k}")
    print(f"[K-Means] Distribución de clusters:")
    print(pd.Series(labels).value_counts().sort_index().to_string())

    return km, labels


# =========================================================
# 4. CLUSTERING JERÁRQUICO (VALIDACIÓN, SOBRE SUBMUESTRA)
# =========================================================
def entrenar_jerarquico_validacion(X, k, n_muestra=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    idx_muestra = rng.choice(len(X), size=min(n_muestra, len(X)), replace=False)
    X_muestra = X.iloc[idx_muestra].reset_index(drop=True)

    modelo_jer = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_jer = modelo_jer.fit_predict(X_muestra)

    print(f"\n[Jerárquico] Entrenado sobre submuestra de {len(X_muestra):,} personas (validación)")

    return labels_jer, idx_muestra


def comparar_kmeans_jerarquico(labels_kmeans_completo, labels_jerarquico, idx_muestra):
    labels_kmeans_muestra = labels_kmeans_completo[idx_muestra]
    ari = adjusted_rand_score(labels_kmeans_muestra, labels_jerarquico)

    print(f"\nAdjusted Rand Index (K-Means vs Jerárquico, misma submuestra): {ari:.4f}")
    if ari > 0.5:
        print("→ Alta concordancia: ambos métodos encuentran una estructura similar")
    elif ari > 0.2:
        print("→ Concordancia moderada: estructura parcialmente consistente")
    else:
        print("→ Baja concordancia: los métodos difieren sustancialmente, revisar K o variables")

    return ari


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")

    print(">>> Construyendo matriz de clustering (versión reducida)...")
    X, df_model = construir_matriz_clustering(df)

    print("\n>>> Determinando K óptimo (codo + silhouette)...")
    tabla_k, k_optimo = determinar_k_optimo(X)

    print(f"\n>>> Entrenando K-Means final con k={k_optimo}...")
    modelo_km, labels_km = entrenar_kmeans(X, k=k_optimo)

    print(f"\n>>> Entrenando jerárquico de validación (submuestra)...")
    labels_jer, idx_muestra = entrenar_jerarquico_validacion(X, k=k_optimo)

    print("\n>>> Comparando K-Means vs Jerárquico...")
    ari = comparar_kmeans_jerarquico(labels_km, labels_jer, idx_muestra)

    df_model["cluster"] = labels_km
    print("\n>>> Muestra de resultado final:")
    print(df_model[["persona_id", "cluster"]].head(10).to_string(index=False))