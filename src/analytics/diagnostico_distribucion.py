import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from cost_drivers import FEATURES

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]

def diagnostico_exploratorio(df, col="gasto_total_anual"):
    """
    Analiza la distribución del target antes de elegir el modelo.
    Imprime estadísticas y genera 3 gráficos de diagnóstico.
    """
    y = df[col].dropna()
    y = y[y > 0]  # GLM Gamma requiere valores estrictamente positivos

    # --- Estadísticas básicas ---
    print(f"N          : {len(y):,}")
    print(f"Media      : {y.mean():,.1f}")
    print(f"Mediana    : {y.median():,.1f}")
    print(f"Asimetría  : {y.skew():.2f}")
    print(f"Curtosis   : {y.kurtosis():.2f}")
    print(f"% ceros    : {(df[col] == 0).mean() * 100:.1f}%")

    # --- Relación varianza-media por cuantiles (elige distribución) ---
    cuantiles = pd.qcut(y, q=10)
    rel = y.groupby(cuantiles).agg(["mean", "var"])
    rel["var/mean²"] = rel["var"] / rel["mean"] ** 2
    rel["var/mean"]  = rel["var"] / rel["mean"]
    print("\nRelación varianza-media por decil:")
    print(rel[["mean", "var/mean", "var/mean²"]].round(2))
    # Si var/mean² ≈ constante → Gamma
    # Si var/mean  ≈ constante → Poisson
    # Si entre ambas           → Tweedie

    # --- Tests de bondad de ajuste ---
    y_log = np.log(y)
    _, p_normal = stats.normaltest(y_log)
    print(f"\nTest normalidad sobre log(gasto): p={p_normal:.4f}")
    print("→ Si p > 0.05, log-normal es candidata")

    # --- Gráficos ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(y, bins=100, color="steelblue", edgecolor="none")
    axes[0].set_title("Distribución gasto (escala original)")
    axes[0].set_xlabel("USD")

    axes[1].hist(np.log1p(y), bins=100, color="steelblue", edgecolor="none")
    axes[1].set_title("Distribución log(gasto)")
    axes[1].set_xlabel("log(USD)")

    stats.probplot(y_log, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot log(gasto) vs Normal")

    plt.tight_layout()
    plt.savefig("diagnostico_target.png", dpi=150)
    plt.close()
    print("\nGráfico guardado: diagnostico_target.png")


def comparar_distribuciones(df, col="gasto_total_anual", peso="peso_muestral"):
    """
    Comparación formal entre distribuciones candidatas via GLM ponderado.
    Usa las mismas features del modelo definitivo para consistencia.
    """
    import statsmodels.api as sm
    import warnings
    from sklearn.model_selection import train_test_split

    df_clean = df[FEATURES + [col, peso]].dropna()
    df_clean = df_clean[df_clean[col] > 0].reset_index(drop=True)

    X = sm.add_constant(df_clean[FEATURES])
    y = df_clean[col].astype(float).values
    w = df_clean[peso].values
    w = w / w.mean()

    idx = df_clean.index.values
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    w_train = w[idx_train]

    n = len(y_train)
    resultados = []

    # --- Gamma ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            glm = sm.GLM(y_train, X_train,
                family=sm.families.Gamma(link=sm.families.links.Log()),
                freq_weights=w_train).fit()
            pred = glm.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - pred)**2))
            k = X_train.shape[1]
            aic = round(2*k - 2*glm.llf, 1)
            bic = round(k*np.log(n) - 2*glm.llf, 1)
            resultados.append({"distribucion": "Gamma", "AIC": aic, "BIC": bic, "RMSE": round(rmse,1)})
            print("[comparar] Gamma OK")
        except Exception as e:
            print(f"[comparar] Gamma falló: {e}")

    # --- Lognormal ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            log_y_train = np.log(y_train)
            wls = sm.WLS(log_y_train, X_train, weights=w_train).fit()
            sigma2 = np.var(wls.resid)
            pred = np.exp(wls.predict(X_test) + sigma2 / 2)
            rmse = np.sqrt(np.mean((y_test - pred)**2))
            k = X_train.shape[1]
            log_lik = -0.5*np.sum(w_train)*(np.log(2*np.pi*sigma2)+1) - np.sum(np.log(y_train))
            aic = round(2*k - 2*log_lik, 1)
            bic = round(k*np.log(n) - 2*log_lik, 1)
            resultados.append({"distribucion": "Lognormal", "AIC": aic, "BIC": bic, "RMSE": round(rmse,1)})
            print("[comparar] Lognormal OK")
        except Exception as e:
            print(f"[comparar] Lognormal falló: {e}")

    # --- Inversa Gaussiana ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            glm = sm.GLM(y_train, X_train,
                family=sm.families.InverseGaussian(link=sm.families.links.Log()),
                freq_weights=w_train).fit()
            pred = glm.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - pred)**2))
            k = X_train.shape[1]
            aic = round(2*k - 2*glm.llf, 1)
            bic = round(k*np.log(n) - 2*glm.llf, 1)
            resultados.append({"distribucion": "Inversa Gaussiana", "AIC": aic, "BIC": bic, "RMSE": round(rmse,1)})
            print("[comparar] Inversa Gaussiana OK")
        except Exception as e:
            print(f"[comparar] Inversa Gaussiana falló: {e}")

    tabla = pd.DataFrame(resultados).sort_values("RMSE")
    print("\nComparación formal de distribuciones (con features, ponderado):")
    print(tabla.to_string(index=False))
    print("\n→ Menor RMSE = mejor predicción fuera de muestra")
    print("→ Menor AIC/BIC = mejor ajuste dentro de muestra")

    return tabla



def estimar_p_tweedie(df, col="gasto_total_anual", peso="peso_muestral"):
    """
    Estima p óptimo de Tweedie por profile likelihood ponderado.
    Incluye ceros — Tweedie los maneja nativamente.
    """
    import statsmodels.api as sm
    import warnings

    df_clean = df[[col, peso]].dropna().reset_index(drop=True)
    y = df_clean[col].astype(float).values
    w = df_clean[peso].values
    w = w / w.mean()
    X = sm.add_constant(np.ones(len(y)))

    p_valores = np.arange(1.1, 2.0, 0.1).round(1)
    resultados = []

    print("\nProfile likelihood ponderado — estimación p óptimo Tweedie:")
    for p in p_valores:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                glm = sm.GLM(y, X,
                    family=sm.families.Tweedie(
                        var_power=p,
                        link=sm.families.links.Log()
                    ),
                    freq_weights=w).fit()
                aic = round(2*2 - 2*glm.llf, 1)
                bic = round(2*np.log(len(y)) - 2*glm.llf, 1)
                resultados.append({"p": p, "log_lik": round(glm.llf, 1), "AIC": aic, "BIC": bic})
                print(f"  p={p:.1f} → log-lik={glm.llf:,.1f}  AIC={aic:,.1f}  BIC={bic:,.1f}")
            except Exception as e:
                print(f"  p={p:.1f} → falló: {e}")

    tabla = pd.DataFrame(resultados)
    p_optimo = tabla.loc[tabla["log_lik"].idxmax(), "p"]
    print(f"\n→ p óptimo por profile likelihood ponderado: {p_optimo}")

    return tabla, p_optimo



if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "meps_2023_clean.csv")
    diagnostico_exploratorio(df)
    print(">>> Iniciando comparar_distribuciones...")
    comparar_distribuciones(df)
    print(">>> Estimando p óptimo Tweedie...")
    estimar_p_tweedie(df)