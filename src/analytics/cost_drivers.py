"""
cost_drivers.py
---------------
GLM de drivers de gasto médico.
Distribución seleccionada formalmente en diagnostico_distribucion.py
y aprobada por el equipo técnico — ver decisiones_tecnicas.md
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

# Decisión formal — no cambiar sin actualizar decisiones_tecnicas.md
DISTRIBUCION_SEVERIDAD = "Gamma"
DISTRIBUCION_PARALELA  = "Lognormal"

FEATURES = [
    "edad", "sexo_femenino", "raza", "estado_civil", "region",
    "años_educacion", "nacido_usa",
    "categoria_pobreza", "ingreso_familiar",
    "tipo_cobertura", "sin_seguro_anual",
    "salud_general", "salud_mental",
    "dx_hipertension", "dx_diabetes", "dx_asma", "dx_cancer",
    "dx_artritis", "dx_cardiopatia", "dx_ictus", "dx_enfisema",
    "n_diagnosticos_unicos", "accidentes_trabajo",
]

def entrenar_glm(df, col="gasto_total_anual", peso="peso_muestral"):
    """
    Entrena GLM Lognormal y Gamma en paralelo con pesos muestrales.
    Evalúa ambos en validación fuera de muestra.
    """
    df_model = df[FEATURES + [col, peso]].dropna()
    df_model = df_model[df_model[col] > 0].reset_index(drop=True)

    X = sm.add_constant(df_model[FEATURES])
    y = df_model[col].astype(float).values
    w = df_model[peso].values
    w = w / w.mean()

    idx = df_model.index.values
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    w_train = w[idx_train]

    resultados = {}

    # --- GLM Lognormal ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            log_y_train = np.log(y_train)
            modelo_ln = sm.WLS(log_y_train, X_train, weights=w_train).fit()
            sigma2 = np.var(modelo_ln.resid)
            pred_ln = np.exp(modelo_ln.predict(X_test) + sigma2 / 2)
            rmse_ln = np.sqrt(np.mean((y_test - pred_ln) ** 2))
            mae_ln  = np.mean(np.abs(y_test - pred_ln))
            resultados["Lognormal"] = {
                "modelo": modelo_ln,
                "rmse": round(rmse_ln, 1),
                "mae": round(mae_ln, 1),
                "aic": round(modelo_ln.aic, 1),
            }
            print(f"[GLM] Lognormal — RMSE={rmse_ln:,.1f}  MAE={mae_ln:,.1f}")
        except Exception as e:
            print(f"[GLM] Lognormal falló: {e}")

    # --- GLM Gamma ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            modelo_gm = sm.GLM(
                y_train, X_train,
                family=sm.families.Gamma(link=sm.families.links.Log()),
                freq_weights=w_train
            ).fit()
            pred_gm = modelo_gm.predict(X_test)
            rmse_gm = np.sqrt(np.mean((y_test - pred_gm) ** 2))
            mae_gm  = np.mean(np.abs(y_test - pred_gm))
            resultados["Gamma"] = {
                "modelo": modelo_gm,
                "rmse": round(rmse_gm, 1),
                "mae": round(mae_gm, 1),
                "aic": round(modelo_gm.aic, 1),
            }
            print(f"[GLM] Gamma     — RMSE={rmse_gm:,.1f}  MAE={mae_gm:,.1f}")
        except Exception as e:
            print(f"[GLM] Gamma falló: {e}")

    # --- Comparación fuera de muestra ---
    print("\nComparación fuera de muestra:")
    print(f"{'Modelo':<12} {'RMSE':>12} {'MAE':>12}")
    print("-" * 38)
    for nombre, res in resultados.items():
        print(f"{nombre:<12} {res['rmse']:>12,.1f} {res['mae']:>12,.1f}")

    ganador = min(resultados, key=lambda k: resultados[k]["rmse"])
    print(f"\n→ Mejor modelo fuera de muestra: {ganador}")
    print("→ Documentar resultado en decisiones_tecnicas.md")

    return resultados


def calcular_vif(df):
    """
    Calcula Variance Inflation Factor para detectar multicolinealidad.
    VIF > 5 = multicolinealidad moderada (revisar)
    VIF > 10 = multicolinealidad severa (eliminar o combinar)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df_model = df[FEATURES].dropna()
    X = sm.add_constant(df_model)

    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    vif = vif[vif["variable"] != "const"]
    vif["alerta"] = vif["VIF"].apply(
        lambda x: "SEVERO" if x > 10 else ("REVISAR" if x > 5 else "OK")
    )
    vif = vif.sort_values("VIF", ascending=False).reset_index(drop=True)

    print("\nVariance Inflation Factor (VIF):")
    print(f"{'Variable':<30} {'VIF':>8} {'Alerta':>10}")
    print("-" * 52)
    for _, row in vif.iterrows():
        print(f"{row['variable']:<30} {row['VIF']:>8.2f} {row['alerta']:>10}")

    print("\n→ VIF > 10 = multicolinealidad severa")
    print("→ VIF > 5  = revisar antes de interpretar coeficientes")

    return vif



def ranking_drivers(resultados):
    """
    Extrae coeficientes de ambos modelos y genera ranking de drivers.
    """
    drivers_por_modelo = {}
    for nombre, res in resultados.items():
        modelo = res["modelo"]
        summary = modelo.summary2().tables[1]
        
        if nombre == "Lognormal":
            summary = modelo.summary2().tables[1]
        else:
            summary = modelo.summary2().tables[1]

        p_col = "P>|z|" if "P>|z|" in summary.columns else "P>|t|"
        drivers = pd.DataFrame({
            "variable":      summary.index,
            "coeficiente":   summary["Coef."].round(4),
            "exp_coef":      np.exp(summary["Coef."]).round(4),
            "p_valor":       summary[p_col].round(4),
            "significativo": summary[p_col] < 0.05,
        })

        drivers = drivers[drivers["variable"] != "const"]
        drivers["impacto_abs"] = drivers["coeficiente"].abs()
        drivers = drivers.sort_values("impacto_abs", ascending=False).reset_index(drop=True)
        drivers_por_modelo[nombre] = drivers

        print(f"\nRanking de drivers — {nombre}:")
        print(f"{'Variable':<30} {'Coef':>8} {'exp(Coef)':>10} {'p-valor':>8} {'Sig':>5}")
        print("-" * 65)
        for _, row in drivers.iterrows():
            sig = "✓" if row["significativo"] else " "
            print(f"{row['variable']:<30} {row['coeficiente']:>8.4f} {row['exp_coef']:>10.4f} {row['p_valor']:>8.4f} {sig:>5}")

    print("\n→ exp(coef) > 1 = aumenta el gasto esperado")
    print("→ exp(coef) < 1 = reduce el gasto esperado")
    print("→ ✓ = estadísticamente significativo (p < 0.05)")

    return drivers_por_modelo

if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")
    print(">>> Entrenando GLM drivers de gasto...")
    resultados = entrenar_glm(df)
    print(">>> Calculando VIF...")
    vif = calcular_vif(df)
    print(">>> Extrayendo ranking de drivers...")
    ranking_drivers(resultados)