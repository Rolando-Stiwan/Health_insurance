# 🏥 HealthRisk360

Pricing de seguro médico — analítica actuarial de siniestralidad en Salud, construida sobre datos longitudinales de MEPS (Medical Expenditure Panel Survey), 2018-2023.

Proyecto de portafolio diseñado para demostrar el ciclo completo de analítica actuarial en Salud: desde ETL y diagnóstico de distribución, pasando por modelado predictivo de frecuencia/severidad, segmentación de riesgo, explicabilidad, gestión de capital, hasta el despliegue de un producto funcional (API + dashboard).

## Stack técnico

- **Datos**: MEPS 2018-2023 (AHRQ), ~156,000 observaciones longitudinales
- **Modelado**: GLM (Gamma, Poisson, Logit, Binomial Negativa) vía `statsmodels`, LightGBM, K-Means
- **Explicabilidad**: SHAP (TreeExplainer + KernelExplainer)
- **Simulación**: Monte Carlo (VaR/CVaR/capital económico), modelo de un factor para riesgo sistémico
- **Despliegue**: FastAPI + Streamlit + Docker
- **Testing**: pytest + GitHub Actions

## Estructura del proyecto

```
healthrisk360/
├── src/
│   ├── ingestion/         # ETL — carga y limpieza de MEPS H251 + H249
│   ├── analytics/         # Drivers de gasto, distribución, tendencias, desviaciones
│   ├── models/             # Frecuencia, severidad, motor de pricing
│   └── segmentation/       # Clustering, detección de riesgo, Monte Carlo, stress test
├── api/                    # FastAPI — endpoints de predicción en producción
├── dashboard/               # Streamlit — dashboard interactivo multi-página
├── tests/                   # Suite de pytest
├── models_artifacts/         # Modelos entrenados y persistidos (no versionado, ver .gitignore)
├── data/                     # Datos crudos y procesados (no versionado, ver .gitignore)
└── decisiones_tecnicas.md    # Bitácora completa de decisiones metodológicas
```

## El proyecto en cinco partes

### 1. Claims & Cost Analytics
- **`cost_drivers.py`**: GLM Gamma de drivers de gasto médico. Distribución seleccionada formalmente comparando Gamma, Lognormal, Inversa Gaussiana y Tweedie por RMSE fuera de muestra.
- **`trend_analysis.py`**: tendencia de gasto y utilización 2018-2023, con GLM controlado por mix demográfico para aislar tendencia real de cambio de composición muestral.
- **`deviation_analysis.py`**: análisis real vs esperado (A/E) por segmento, con intervalos de confianza vía bootstrap ponderado.

### 2. Predictive Modeling
- **`claim_freq.py`**: modelo de frecuencia de utilización (Poisson vs Binomial Negativa, comparación formal por RMSE y diagnóstico de sobredispersión).
- **`claim_sev.py`**: modelo de severidad (GLM Gamma), con arquitectura de entrenamiento/inferencia separada (persistencia vía `joblib`).
- **`pricing_engine.py`**: motor de pricing — modelo de dos partes (P(gasto>0) × severidad) + credibilidad de Bühlmann-Straub por segmento.

### 3. Risk Segmentation
- **`clustering.py`**: segmentación K-Means (k=2, seleccionado por silhouette score) sobre variables clínicas/utilización, validado contra clustering jerárquico.
- **`segment_profiles.py`**: caracterización demográfica/clínica y de costo por segmento.
- **`risk_detection.py`**: detección de alto riesgo — Isolation Forest (anomalías, no supervisado) + LightGBM (clasificación supervisada, AUC-ROC 0.85).

### 4. Explainability & Business Insights
- **`explainability.py`**: SHAP global y local para los modelos de severidad y riesgo.
- **`montecarlo.py`**: simulación Monte Carlo de pérdida agregada del portafolio — VaR, CVaR, capital económico.
- **`stress_test.py`**: pruebas de estrés — escenario de pandemia (riesgo sistémico correlacionado, modelo de un factor) y catástrofe regional (choque localizado).

### 5. Deployment
- **API REST (FastAPI)**: endpoints `/riesgo`, `/severidad`, `/pricing`, `/stress-test`, con modelos cargados una sola vez al inicio (arquitectura training/inference separada).
- **Dashboard (Streamlit)**: 6 páginas — Desviaciones, Drivers, Segmentos, Pricing (conectado a la API), Stress Test (conectado a la API), Explainability (SHAP).
- **Tests**: 28 tests de pytest cubriendo ETL, modelos y API.
- **CI**: GitHub Actions valida instalabilidad del paquete en cada push (los datos y modelos no se versionan, ver nota abajo).
- **Docker**: containerización de API y dashboard para despliegue en Render/Railway.

## Decisiones técnicas destacadas

Todo el razonamiento detrás de cada decisión metodológica —incluyendo bugs detectados y corregidos durante el desarrollo— está documentado en [`decisiones_tecnicas.md`](decisiones_tecnicas.md). Algunos puntos destacados:

- **Modelo de dos partes** (P(gasto>0) × severidad) en vez de frecuencia×severidad ingenuo — se detectó y corrigió un error de diseño que producía sobreestimación de ~120x al combinar conteo de eventos de utilización con severidad anual total.
- **Corrección de un bug de datos silencioso**: un archivo de corte anual desactualizado introducía ~29% de NaN espurios en una variable clave, sesgando sistemáticamente la muestra de varios modelos. Detectado, diagnosticado y corregido con trazabilidad completa.
- **Limitación de calibración conocida**: el modelo de severidad sobreestima sistemáticamente el segmento de mayor riesgo (~77% en el cluster de alto costo), confirmada de forma independiente por tres métodos distintos (análisis de desviaciones, especificación categórica, perfiles de cluster). Documentada como prioridad de iteración futura, no oculta.
- **Riesgo sistémico vs idiosincrático**: el capital económico bajo escenario de pandemia (riesgo correlacionado) es ~5.6x mayor que el baseline, pese a que la pérdida esperada solo sube ~10% — evidencia cuantitativa de por qué la correlación importa en gestión de capital de seguros de salud.

## Cómo correr el proyecto

### Instalación
```bash
pip install -e .
```

### Pipeline de datos y modelos (orden de ejecución)
```bash
python src/ingestion/etl.py
python src/analytics/diagnostico_distribucion.py
python src/analytics/cost_drivers.py
python src/analytics/trend_analysis.py
python src/analytics/deviation_analysis.py
python src/models/claim_freq.py
python src/models/claim_sev.py
python src/models/pricing_engine.py
python src/segmentation/risk_detection.py
```

### Tests
```bash
pytest -v
```

### API
```bash
uvicorn api.main:app --reload
# Documentación interactiva: http://127.0.0.1:8000/docs
```

### Dashboard
```bash
streamlit run dashboard/app.py
```

## Despliegue con Docker

La API y el dashboard se empaquetan juntos en un único contenedor, por
decisión deliberada (no la arquitectura "ideal" de microservicios, sino
la más pragmática para este contexto):

- Render y Railway, en su capa gratuita, exponen típicamente un solo
  puerto público por servicio — desplegar API y dashboard como
  contenedores separados requeriría 2 servicios (con más fricción de
  configuración y, en algunos planes, costo adicional).
- Dentro del contenedor, la API corre en el puerto 8000 (uso interno,
  no expuesto), y el dashboard Streamlit se sirve en el puerto público
  asignado por la plataforma. Las páginas de Pricing y Stress Test del
  dashboard llaman a la API vía `localhost:8000` — funciona sin
  configuración adicional porque ambos procesos comparten el mismo
  contenedor.

En un entorno de producción real, con tráfico y necesidad de escalar
cada servicio de forma independiente, la arquitectura correcta sería
contenedores separados orquestados con docker-compose o Kubernetes —
aquí se prioriza simplicidad de despliegue sobre separación de
responsabilidades, una decisión consciente para el alcance de este
proyecto.

Nota técnica: las versiones de FastAPI, Starlette, Uvicorn y Streamlit
están fijadas explícitamente en `pyproject.toml` (en vez de dejarse sin
pin) tras detectar un conflicto de compatibilidad entre Starlette y
Streamlit al reconstruir el entorno desde cero dentro del contenedor
— una versión no fijada resolvía a una combinación incompatible que
nunca aparecía en el entorno de desarrollo local (ya instalado con
versiones compatibles de antes). Buena práctica general: fijar
versiones exactas evita que "funciona en mi máquina" se rompa al
reconstruir el entorno en otro lugar.

### Construir y correr localmente
```bash
docker build -t healthrisk360 .
docker run -p 8501:8501 -e PORT=8501 healthrisk360
```
Luego abre `http://localhost:8501`.

### Desplegar en Render/Railway
Ambas plataformas detectan automáticamente el `Dockerfile` al conectar
el repositorio de GitHub. Configurar la variable de entorno `PORT`
según lo requiera la plataforma (Render/Railway la inyectan
automáticamente en la mayoría de los casos).


### Limitación de memoria en el despliegue gratuito
El endpoint /stress-test corre sobre una submuestra de 2,000 personas 
(en vez de las ~15,000 del dataset completo) para mantenerse dentro del límite de 512MB 
de RAM del tier gratuito de Render. La metodología completa (con el 
dataset completo, 10,000 simulaciones) está documentada y validada en 
decisiones_tecnicas.md y puede reproducirse localmente sin esta 
restricción.


## Nota sobre datos y modelos

Los archivos de datos MEPS crudos/procesados y los modelos entrenados (`models_artifacts/`) no se versionan en este repositorio por tamaño y por licenciamiento de los datos originales (públicos de AHRQ, descargables directamente desde [meps.ahrq.gov](https://meps.ahrq.gov)). Como consecuencia, la suite de tests en CI (GitHub Actions) valida instalabilidad y estructura del paquete, pero la mayoría de tests funcionales se ejecutan localmente, donde sí existen los datos y modelos.

## Contexto del proyecto

Construido como pieza de portafolio para roles de analítica actuarial en Salud, con foco en las competencias explícitamente requeridas: análisis de drivers de coste y desviaciones, detección de tendencias en consumo sanitario, modelos predictivos de frecuencia y coste, y traducción de resultados técnicos en insights accionables para negocio.

**Autor**: Rolando Stiwan — [GitHub](https://github.com/Rolando-Stiwan)
