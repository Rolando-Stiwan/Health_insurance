# Decisiones Técnicas — HealthRisk360

## Dataset
- Fuente: MEPS 2023 (Medical Expenditure Panel Survey), AHRQ
- Archivos: H251 (Full Year Consolidated) + H249 (Medical Conditions)
- Años disponibles: 2018-2023 (6 años longitudinales)
- Población: ~18,000-20,000 personas por año, representativas de la población civil no institucionalizada de EE.UU.
- Los pesos muestrales (PERWTyyF) se normalizan (w / w.mean()) antes de usarlos en GLMs para evitar inflación de la log-verosimilitud.

| Año  | H251 (Consolidated) | H249 (Medical Conditions) |
|------|---------------------|---------------------------|
| 2018 | h209.xlsx           | h207.xlsx                 |
| 2019 | h216.xlsx           | h214.xlsx                 |
| 2020 | h224.xlsx           | h222.xlsx                 |
| 2021 | h233.xlsx           | h231.xlsx                 |
| 2022 | h243.xlsx           | h241.xlsx                 |
| 2023 | h251.xlsx           | h249.xlsx                 |

## ETL Longitudinal — Estrategia de integración multi-año

### Estrategia: contrato flexible con fillna (Opción B)

**Justificación:**
1. Descartar columnas ausentes en años anteriores implica perder información
   disponible en años recientes — viola el principio de máximo aprovechamiento
   de datos en actuarial.
2. Los modelos de tendencia temporal requieren el mayor número de observaciones
   posible — eliminar variables reduce el poder estadístico innecesariamente.
3. En la práctica, aseguradoras que integran datos históricos de múltiples
   fuentes usan NaN para períodos donde una variable no existía, documentando
   la fecha de disponibilidad.
4. Los módulos downstream (modelado, segmentación, pricing) ya manejan NaN
   via dropna() selectivo — no requieren cambios.

### Columnas con disponibilidad parcial en H249
| Columna | Disponible desde |
|---------|-----------------|
| CCSR4X  | 2022            |
| ERCOND, IPCOND, OPCOND, OBCOND, RXCOND, HHCOND | 2021 |

### Implicación para análisis
- Tendencias 2018-2023: usan columnas estables en todos los años
- Análisis de utilización por condición: solo 2021-2023
- Clasificación clínica CCSR4X: solo 2022-2023

## Tratamiento de ceros en gasto_total_anual
- ~14% de la muestra tiene gasto = 0 (asegurados sin reclamaciones en el año)
- Los ceros son observaciones válidas, no errores
- Decisión: modelo de dos partes — Parte 1 (logístico): probabilidad de tener gasto; Parte 2 (severidad): cuánto, dado que hay gasto
- Tweedie se evalúa con todos los datos incluyendo ceros por ser la única distribución candidata que los maneja nativamente

## Selección de distribución para severidad (gasto_total_anual > 0)

### Metodología
Comparación formal via GLM ponderado (peso_muestral normalizado) usando
las features del modelo definitivo y evaluación fuera de muestra (80/20,
random_state=42).

- Candidatas evaluadas: Lognormal, Gamma, Inversa Gaussiana, Tweedie
- Tweedie: p óptimo estimado por profile likelihood ponderado → p=1.9 (converge a Gamma, descartada)
- Inversa Gaussiana: descartada formalmente

### Resultados con features (decisión definitiva)

| Distribución | AIC | BIC | RMSE fuera de muestra |
|---|---|---|---|
| Lognormal | 199,853 | 200,027 | 241,770 |
| Gamma | 204,913 | 205,086 | 92,347 |
| Inversa Gaussiana | 210,157 | 210,330 | inestable (~2.9B, descartada) |

### Tensión AIC vs RMSE
- Lognormal gana en AIC — mejor ajuste dentro de muestra
- Gamma gana en RMSE — mejor predicción fuera de muestra por factor de 2.7x
- Inversa Gaussiana descartada definitivamente

### Decisión final
**Distribución seleccionada: Gamma**

Justificación:
1. El objetivo del modelo es pricing — predecir gasto futuro de asegurados
2. RMSE fuera de muestra es la métrica relevante para pricing, no AIC
3. Gamma supera a Lognormal en predicción fuera de muestra por factor de 2.8x
4. Gamma es el estándar de la industria aseguradora para severidad de gastos
5. Tweedie con p=1.9 converge a Gamma — confirma la elección

Lognormal se mantiene como modelo paralelo para monitoreo — si en revisiones
futuras el RMSE converge, se reevalúa la decisión en comité técnico.



## Multicolinealidad y selección de features

### VIF calculado sobre features del modelo
Todos los VIF dentro del rango aceptable (< 5) tras eliminar variables problemáticas.
VIF máximo: tipo_cobertura = 2.71.

### Variables eliminadas por VIF severo
| Variable | VIF | Razón |
|---|---|---|
| n_sistemas_afectados | 76 | Redundante con n_diagnosticos_unicos |
| lesiones | 62 | Capturada por accidentes_trabajo |

### Features definitivas del modelo
edad, sexo_femenino, raza, estado_civil, region, años_educacion, nacido_usa,
categoria_pobreza, ingreso_familiar, tipo_cobertura, sin_seguro_anual,
salud_general, salud_mental, dx_hipertension, dx_diabetes, dx_asma, dx_cancer,
dx_artritis, dx_cardiopatia, dx_ictus, dx_enfisema, n_diagnosticos_unicos,
accidentes_trabajo

## Drivers de gasto — hallazgos del GLM Gamma

RMSE Gamma: 92,347. Gamma se mantiene ~2.6x mejor que Lognormal fuera de
muestra (RMSE Lognormal = 241,770), confirmando la decisión de
distribución tomada en diagnostico_distribucion.py.

### Drivers significativos que suben el gasto
| Variable | exp(coef) | Interpretación |
|---|---|---|
| sin_seguro_anual | 1.47 | Sin seguro sube el gasto 47% |
| dx_cancer | 1.35 | Cáncer sube el gasto 35% |
| salud_general | 1.32 | Cada punto peor sube 32% |
| dx_diabetes | 1.27 | Diabetes sube 27% |
| n_diagnosticos_unicos | 1.24 | Cada diagnóstico adicional sube 24% |
| accidentes_trabajo | 1.05 | Efecto pequeño pero significativo |

### Drivers significativos que bajan el gasto
| Variable | exp(coef) | Interpretación |
|---|---|---|
| tipo_cobertura | 0.80 | Mejor cobertura reduce el gasto |
| dx_hipertension | 0.86 | Efecto tratamiento — pacientes controlados |
| region | 0.96 | Variación geográfica significativa |

### Nota sobre dx_hipertension
Coeficiente negativo confirmado como efecto de tratamiento — no es error
del modelo. Pacientes con hipertensión diagnosticada están bajo control
médico activo, lo que reduce su gasto relativo vs pacientes no
diagnosticados. VIF de hipertensión = 1.51 descarta multicolinealidad
como causa.

### Nota sobre dx_cardiopatia
No resulta estadísticamente significativo (exp(coef)=1.14, p=0.166).
Los demás drivers de la tabla sí son significativos y consistentes con
la literatura clínica.

## Limitaciones documentadas
- BIC de statsmodels con freq_weights produce valores negativos (bug
  conocido) — se recalcula manualmente con k*log(n) - 2*llf
- Análisis de utilización por condición limitado a 2021-2023 por
  disponibilidad de variables en H249



## Análisis de desviaciones real/esperado (deviation_analysis.py)

### Metodología
- GLM Gamma reentrenado sobre split 80/20 (random_state=42, mismo de cost_drivers.py)
- Predicciones generadas para todo el dataset con gasto > 0 (n=12,817)
- Ratio A/E (Actual/Esperado) ponderado por peso_muestral, con IC 95% vía
  bootstrap ponderado (500 remuestreos)
- Segmentos: region, tipo_cobertura (INSCOV), categoria_pobreza (POVCAT),
  salud_general (RTHLTH), sin_seguro_anual (UNINS)

### Codificación de variables (MEPS 2023 codebook, confirmado)
- POVCAT23: 1=Pobre, 2=Casi pobre, 3=Ingreso bajo, 4=Ingreso medio, 5=Ingreso alto
- INSCOV23: 1=Privado, 2=Público solamente, 3=Sin seguro
- RTHLTH53: 1=Excelente, 2=Muy buena, 3=Buena, 4=Regular, 5=Mala
- UNINS23: 1=Sin seguro todo el año, 2=Con seguro
- REGION23: 1=Noreste, 2=Medio Oeste, 3=Sur, 4=Oeste

### Hallazgo principal: sesgo sistemático de sobreestimación
El modelo sobreestima el gasto en casi todos los segmentos (ratio A/E < 1),
con sesgo creciente y monótono en los extremos de mayor severidad esperada.

**Por región** — las 4 regiones muestran desviación significativa:
| Región | ratio A/E |
|---|---|
| Noreste | 0.724 |
| Oeste | 0.695 |
| Sur | 0.669 |
| Medio Oeste | 0.610 |

**Por tipo de cobertura** — el sesgo más marcado del análisis:
| Cobertura | ratio A/E | Significativo |
|---|---|---|
| Sin seguro | 1.093 | No (n pequeño) |
| Privado | 0.763 | Sí |
| Público solamente | 0.532 | Sí |

El modelo sobreestima casi a la mitad el gasto de la población con
cobertura pública (Medicare/Medicaid) — el sesgo más severo de todo
el análisis.

**Por salud general** — patrón perfectamente monótono con la severidad:
| Salud general | ratio A/E |
|---|---|
| Excelente | 1.134 |
| Muy buena | 0.891 |
| Buena | 0.714 |
| Regular | 0.488 |
| Mala | 0.436 |

Evidencia clara de mala especificación funcional: la relación entre
salud_general y gasto no es lineal en escala log, y el modelo actual
(lineal-log) no la captura, especialmente en el extremo de peor salud.

**Por categoría de pobreza**: sesgo significativo en las 5 categorías,
más marcado en niveles bajo-medio (ratio ~0.53-0.60) que en el extremo
de mayor ingreso (ratio 0.694-0.780).

**Por sin_seguro_anual**: la población con seguro todo el año muestra
sesgo significativo (ratio 0.668); sin seguro no es significativo por
tamaño de muestra pequeño (n=519).

### Interpretación y limitación metodológica
Los segmentos analizados ya son features del modelo Gamma. El GLM
minimiza error global, no por subgrupo, así que estos ratios reflejan
calibración del modelo (qué tan bien ajusta cada subgrupo), no
"eficiencia de proveedor" u otra fuente de desviación externa al
modelo — eso requeriría datos a nivel de proveedor que MEPS no incluye.

### Implicación para el modelo de pricing
El sesgo más severo y accionable es tipo_cobertura=público (-47%) y
salud_general en el extremo malo (-56%). Ambos apuntan a la misma
causa: relaciones no lineales que el GLM lineal-log actual no captura.
Candidatos para mejora en iteraciones futuras: splines en salud_general,
o tratamiento categórico en vez de continuo.



## Modelo de frecuencia (claim_freq.py)

### Target
n_eventos_utilizacion = visitas_ambulatorias + visitas_outpatient +
visitas_urgencias + noches_hospital. Se excluye total_recetas (evento
de dispensación farmacéutica, no encuentro médico). Todas las
observaciones se incluyen, incluyendo frecuencia=0.

### Comparación formal Poisson vs Binomial Negativa
| Modelo | RMSE | MAE | AIC |
|---|---|---|---|
| Poisson | 16.549 | 7.250 | 142,845.7 |
| Binomial Negativa | 290.344 | 20.409 | 66,618.1 |

Decisión: Poisson, por mismo criterio de RMSE fuera de muestra usado en
severidad (diagnostico_distribucion.py). NB gana en AIC (mejor ajuste
dentro de muestra al corregir sobredispersión) pero resulta inestable
fuera de muestra en esta especificación.

### Diagnóstico de sobredispersión
Ratio varianza/media: 28.29. Estadístico de dispersión (Pearson): 14.68.
Evidencia clara de sobredispersión — Poisson subestima la variabilidad
real. Limitación conocida y documentada, no corregida en esta iteración
porque el criterio de RMSE prioriza la decisión de modelo final.

### Drivers de frecuencia — hallazgo principal
| Variable | exp(coef) | Interpretación |
|---|---|---|
| sin_seguro_anual | 2.18 | Ver nota abajo — no interpretar como causal |
| dx_enfisema | 0.74 | Reduce frecuencia (posible subutilización) |
| sexo_femenino | 1.20 | Mujeres con +20% frecuencia |
| n_diagnosticos_unicos | 1.16 | Cada diagnóstico adicional sube 16% |
| dx_hipertension | 0.88 | Reduce frecuencia (pacientes controlados) |
| dx_cancer | 1.13 | Sube frecuencia 13% |

### Nota sobre sin_seguro_anual — interpretación cuidadosa requerida
sin_seguro_anual es, con amplio margen, el driver más fuerte del modelo
(+118% frecuencia esperada, controlando por demografía y comorbilidades).
Esto es contraintuitivo a primera vista: la literatura de salud pública
documenta típicamente que la falta de seguro reduce el uso del sistema
por barrera de costo.

Explicación más plausible: el diseño observacional del estudio no
permite establecer causalidad. Es posible que la pérdida de seguro sea
consecuencia, no causa, de episodios de salud costosos (pérdida de
empleo por enfermedad, agotamiento de recursos). También es consistente
con un patrón de atención reactiva — sin cobertura para atención
preventiva regular, la persona termina usando urgencias/hospitalización
(que no pueden negarse por ley) en vez de consultas ambulatorias más
baratas y tempranas.

Este hallazgo es coherente con cost_drivers.py, donde sin_seguro_anual
también aumenta la severidad (+47% gasto). Juntos sugieren un patrón de
atención tardía y reactiva en la población sin seguro, no simplemente
"más uso del sistema" sin contexto.

No se reporta como relación causal sin_seguro → mayor uso; se reporta
como asociación fuerte y bidireccional plausible, relevante para
pricing pero no para políticas de intervención sin estudio causal
adicional.

### Limitación
Sobredispersión no corregida (Poisson elegido por RMSE, no por ajuste
de varianza). Iteraciones futuras podrían explorar Binomial Negativa
con especificación distinta, o Tweedie compuesto para frecuencia.




## Modelo de severidad — arquitectura de producción (claim_sev.py)

Segundo módulo de Parte 2. Reutiliza la distribución Gamma ya
seleccionada formalmente (diagnostico_distribucion.py) y las FEATURES
de cost_drivers.py, pero separa entrenamiento de inferencia:

- entrenar_y_guardar_severidad(): entrena y persiste el modelo
  (models_artifacts/severity_gamma.joblib) junto con metadata
  versionada (fecha, RMSE, features, random_state) en JSON.
- cargar_modelo_severidad() / predecir_severidad(): cargan el
  artefacto ya entrenado, sin reentrenar — usadas por
  pricing_engine.py y, en Parte 5, por el endpoint /pricing de FastAPI.

Decisión de diseño: separación training/inference es estándar en
MLOps y evita reentrenar el modelo en cada llamada de producción,
relevante para latencia de API y para trazabilidad/gobernanza de
modelos de pricing (versionado con fecha y métricas).

RMSE test: 92,347.4 (idéntico a cost_drivers.py y deviation_analysis.py,
confirma consistencia del pipeline sobre los datos corregidos).






## Motor de pricing (pricing_engine.py)

### Arquitectura
Modelo de dos partes: P(gasto>0) [logit, entrenado aquí] × E[gasto|gasto>0]
[Gamma, claim_sev.py]. Prima pura calculada a nivel individual, luego
ajustada por credibilidad de Bühlmann-Straub (método de momentos) por
tipo_cobertura — el segmento con mayor impacto de negocio detectado en
deviation_analysis.py.

Nota de diseño: claim_freq.py (conteo de eventos de utilización) no se
usa como componente de "frecuencia" en esta fórmula — representa
utilización, no probabilidad de gasto, y combinarlo directamente con
severidad_total_anual produce doble conteo (se detectó y corrigió
durante el desarrollo: prima pura inicial de $1.15M vs gasto real de
$9,535, error de ~120x).

### Modelo de probabilidad P(gasto>0)
GLM Logit — Accuracy=0.907, Brier=0.0569, AIC=4,049.5.

### Dos enfoques de credibilidad (comparación deliberada)
- MODELO (producción): credibilidad aplicada sobre prima_pura (predicción)
- EXPERIENCIA CRUDA (validación): credibilidad aplicada sobre gasto_total_anual real

| Tipo cobertura | Prima modelo | Prima experiencia cruda | Diferencia |
|---|---|---|---|
| Privado | $11,054 | $8,426 | +31.2% |
| Público | $19,264 | $10,299 | +87.0% |
| Sin seguro | $1,909 | $1,368 | +39.6% |

### Hallazgo crítico: el modelo sobreestima sistemáticamente, más severo en cobertura pública
Consistente con el diagnóstico ya documentado en deviation_analysis.py
(ratio A/E de tipo_cobertura=público = 0.532, el más severo del
análisis). La prima de producción calculada por el modelo sobreestimaría
el precio real hasta en 87% para la población con cobertura pública.

### Estado del pipeline
La arquitectura (modelo de dos partes, credibilidad de Bühlmann,
persistencia de modelos, separación training/inference) está completa
y funcional. El modelo Gamma de severidad subyacente requiere mejora de
especificación (splines en salud_general, revisión de interacción con
tipo_cobertura) antes de considerar esta prima apta para uso en
producción real. Esto se identifica como próximo paso de iteración,
no como limitación oculta.

## Nota sobre pricing_engine.py — coeficientes del modelo logit P(gasto>0)

edad no es estadísticamente significativa (coef=0.0042, p=0.215) — el
modelo apenas usa la edad para predecir probabilidad de gasto.
n_diagnosticos_unicos domina con coeficiente 5.80 (p<0.0001, exp(coef)≈330),
consistente con su dominancia ya observada en severidad (SHAP) y en
LightGBM (importancia de variables). Esto explica el comportamiento
observado en el dashboard de Pricing: activar cualquier diagnóstico
satura rápidamente la probabilidad predicha hacia 100%, mientras que
variaciones de edad solo, sin cambios en diagnósticos, apenas mueven
la probabilidad.

__________________________________________________________________________________________



## Segmentación de riesgo — clustering (clustering.py, segment_profiles.py)

### Metodología
Clustering K-Means sobre variables de riesgo clínico/utilización 
(edad, salud_general, salud_mental, n_diagnosticos_unicos, tipo_cobertura,
categoria_pobreza, sin_seguro_anual, diagnósticos dx_*), excluyendo
variables demográficas de bajo valor discriminativo (raza, estado_civil,
region, sexo_femenino, años_educacion, accidentes_trabajo) tras detectar
silhouette muy bajo (0.14) y ARI ~0.06 (prácticamente aleatorio) vs
clustering jerárquico con el set completo de 51 columnas codificadas —
síntoma de curse of dimensionality por exceso de categóricas dispersas.

Con el set reducido (21 columnas): silhouette=0.218 (k=2), ARI=0.468
(concordancia moderada K-Means vs jerárquico Ward sobre submuestra de
2,000 personas). K=2 elegido por máximo silhouette; el codo de inercia
no muestra quiebre claro en ningún k, consistente con una población de
riesgo distribuida como espectro continuo más que en arquetipos
discretos — hallazgo esperable en datos de salud humana.

No se incluyó ninguna salida de modelo de costo (prima_pura, gasto)
como insumo del clustering, para evitar circularidad metodológica.

### Perfiles resultantes
| Cluster | n | Edad media | Diagnósticos media | % Hipertensión | Gasto real medio |
|---|---|---|---|---|---|
| 0 (bajo riesgo) | 9,016 (61%) | 40.9 | 1.6 | 16% | $4,073 |
| 1 (alto riesgo) | 5,867 (39%) | 63.9 | 6.6 | 66% | $17,594 |

Diferencia de gasto real: 4.3x entre segmentos, con perfiles clínicos
claramente diferenciados e interpretables.

### Hallazgo de calibración — confirma limitación ya documentada
| Cluster | Gasto real medio | Prima pura del modelo | Ratio modelo/real |
|---|---|---|---|
| 0 | $4,073 | $3,786 | 0.93 |
| 1 | $17,594 | $31,125 | 1.77 |

El modelo de pricing sobreestima severamente (+77%) el costo del
segmento de alto riesgo. Consistente con el sesgo ya identificado en
deviation_analysis.py (mala especificación de salud_general en el
extremo de alta severidad). Esta es la tercera confirmación independiente
del mismo hallazgo (por variable individual, por corrección con dummies,
y ahora por cluster agregado) — refuerza que es una limitación real y
sistemática del modelo actual, prioritaria para una futura iteración.

### Corrección de bug — desalineación de índices
Se detectó y corrigió un bug de desalineación de índices en
segment_profiles.py: construir_matriz_clustering() reseteaba el índice
tras dropna(), causando que variables reincorporadas después (sexo,
gasto, peso muestral) no correspondieran a la persona correcta. Señal
de detección: %mujeres idéntico (50.5%) en ambos clusters, 
estadísticamente improbable. Corregido preservando el índice original 
de principio a fin.

_______________________________________________________________________________________

## Detección de alto riesgo (risk_detection.py)

### Isolation Forest — detección de anomalías (no supervisado)
contamination=0.20 (igualada a la prevalencia de alto_coste para
comparación). Detectó 2,946 anomalías (20.0% de la muestra).

Solapamiento con alto_coste real: 45.3%. Solapamiento parcial esperado
por diseño — anomalía y alto costo son conceptos relacionados pero
distintos. 1,611 personas anómalas no son de alto costo (candidatas a
revisión de calidad de datos o perfiles atípicos no costosos); 2,095
de alto costo no son anómalas (costosas pero con perfil "normal",
extremo esperado de un patrón conocido).

### LightGBM — clasificación supervisada de alto riesgo
Target: alto_coste (top 20% de gasto_total_anual, ya definido en
etl.py). scale_pos_weight=3.29 para compensar desbalance de clases.

AUC-ROC: 0.8501 — buena capacidad discriminativa, superior a los GLMs
lineales de severidad en este problema de clasificación.

Umbral de decisión: se probó formalmente si el umbral por defecto (0.5)
era subóptimo dado el reponderamiento de clases. Resultado: mejora
marginal con umbral=0.579 (F1: 0.593 → 0.598, +0.8%; precision: 0.49 →
0.535; recall: 0.75 → 0.676). El umbral por defecto ya estaba cerca del
óptimo — se documenta la verificación aunque la mejora fue pequeña.
Elección final de umbral (0.5 vs 0.579) es una decisión de negocio
(costo relativo de falsos positivos vs falsos negativos en gestión
médica), no puramente estadística.

### Importancia de variables — contraste con los GLMs
| Variable | Importancia |
|---|---|
| ingreso_familiar | 21.8% |
| edad | 16.5% |
| n_diagnosticos_unicos | 10.9% |
| años_educacion | 6.5% |
| salud_general | 6.4% |

A diferencia de los GLMs de severidad (donde dx_cancer, salud_general y
sin_seguro_anual dominaban), LightGBM prioriza ingreso_familiar y edad
— sugiere que el modelo basado en árboles captura patrones no lineales
o de interacción entre variables socioeconómicas que los GLMs lineales
no modelan explícitamente. Contraste interesante entre enfoques, no
necesariamente una contradicción.

______________________________________________________________________________

## Explicabilidad — SHAP (explainability.py)

### Metodología
- LightGBM (alto riesgo): TreeExplainer, sobre muestra de 2,000 personas
- GLM Gamma (severidad): KernelExplainer (model-agnostic, sin explainer
  nativo para GLM statsmodels), sobre muestra de 200 personas / 100
  background — ejecución rápida en la práctica (~5 segundos)

### Hallazgo principal: n_diagnosticos_unicos domina en ambos modelos
SHAP global coincide entre LightGBM y GLM Gamma: n_diagnosticos_unicos
es la variable más importante en ambos, con una diferencia de ~10x
sobre la segunda variable — validación cruzada entre dos modelos
independientes (árboles vs GLM lineal), para dos targets distintos
(alto_coste vs severidad).

Nota importante: esto no contradice el ranking de coeficientes de
cost_drivers.py (donde sin_seguro_anual, dx_cancer y salud_general
tenían mayor exp(coef)). SHAP mide contribución real a la predicción,
que depende del coeficiente Y del rango de variación de cada variable
en los datos. n_diagnosticos_unicos varía en un rango amplio (0-10+),
mientras que variables binarias (sin_seguro_anual) solo pueden mover
la predicción en una cantidad fija — por eso n_diagnosticos_unicos
domina en SHAP aunque su coeficiente individual sea menor.

### Ranking SHAP global — LightGBM (alto riesgo)
n_diagnosticos_unicos, salud_general, ingreso_familiar, dx_diabetes, edad

### Ranking SHAP global — GLM Gamma (severidad)
n_diagnosticos_unicos, salud_general, accidentes_trabajo, dx_cancer,
tipo_cobertura

### Ejemplo de explicación local (severidad)
Persona con 10 diagnósticos, salud general excelente, diabetes,
cobertura pública: predicción $21,820 (base $17,077 + ajustes). El
factor que más sube la predicción es n_diagnosticos_unicos (+$11,383);
el que más la reduce es salud_general=Excelente (-$9,950), pese a la
alta comorbilidad — el modelo captura que la autopercepción de buena
salud atenúa el gasto esperado incluso con múltiples diagnósticos.


_______________________________________________________________

## Simulación Monte Carlo — VaR, CVaR, capital económico (montecarlo.py)

### Metodología
Por persona: Bernoulli(prob_gasto) x Gamma(shape, scale), con shape y
scale derivados de la severidad media predicha (claim_sev.py) y la
dispersión real del GLM Gamma (model.scale = 4.04). 10,000 simulaciones,
vectorizadas en bloques de 500. Portafolio = muestra real de 14,728
personas (no expandida por peso_muestral — capital económico se calcula
sobre un portafolio concreto de asegurados, no sobre la población
nacional representada por la encuesta).

Independencia asumida entre personas (riesgo idiosincrático). Escenarios
correlacionados (pandemia, catástrofe) se tratan por separado en
stress_test.py.

### Chequeo de sanidad
Pérdida esperada simulada ($229,547,294) vs suma analítica de primas
puras ($229,567,640): diferencia de -0.01% — confirma convergencia
correcta de la simulación.

### Resultados
| Nivel de confianza | VaR | CVaR | Capital económico |
|---|---|---|---|
| 95% | $269.4M | $285.0M | $39.8M |
| 99% | $293.9M | $310.6M | $64.4M |

Capital económico al 99% representa ~28% de la pérdida esperada del
portafolio — ratio razonable dada la asimetría (colas largas) típica
de datos de gasto médico.

### Limitación e interpretación
Estas cifras son específicas del portafolio sintético (14,728 personas,
MEPS 2023) y de los modelos ya entrenados — incluyen la limitación de
calibración en la cola de alto riesgo ya documentada en
deviation_analysis.py y segment_profiles.py (el modelo sobreestima el
segmento de alto riesgo en ~77%). No son directamente aplicables a una
cartera real de Aegon sin recalibración con datos propios de la
aseguradora; son una demostración metodológica completa del pipeline
de capital económico.

_________________________________________________________________________


## Pruebas de estrés — pandemia y catástrofe regional (stress_test.py)

### Metodología
- PANDEMIA: modelo de un factor (cópula gaussiana, mismo principio que
  el modelo de Vasicek usado en CreditRiskLab para riesgo de crédito
  correlacionado). Factor sistémico Z común al portafolio en cada
  simulación, correlación rho=0.15 (supuesto de calibración, no
  estimado de los datos — MEPS no tiene información de correlación
  sistémica entre personas). Multiplicador de severidad sistémico=0.4.
- CATÁSTROFE REGIONAL: choque de severidad localizado (multiplicador
  2.5x) a la región 3 (Sur), 5,701 personas afectadas (38.7% del
  portafolio) — sin correlación sistémica, choque determinístico y
  concentrado geográficamente.

### Resultados comparativos (capital económico al 99%)
| Escenario | Pérdida esperada | Capital económico | Incremento vs baseline |
|---|---|---|---|
| Baseline | $229.5M | $64.4M | — |
| Pandemia | $252.1M | $357.9M | +455.9% |
| Catástrofe regional | $345.1M | $84.1M | +30.6% |

### Hallazgo principal
El riesgo de pandemia (correlacionado/sistémico) requiere ~5.6x más
capital económico que el baseline, pese a que la pérdida esperada
apenas sube 9.8% — la correlación infla desproporcionadamente la cola
de la distribución, no la media. La catástrofe regional (choque
determinístico, localizado, sin correlación sistémica) sube mucho más
la pérdida esperada (+50.4%) pero el capital económico adicional es
mucho menor (+30.6%).

Conclusión de negocio: el riesgo sistémico/correlacionado (pandemias,
crisis de salud pública) es fundamentalmente más peligroso para la
solvencia de una aseguradora que un choque localizado de magnitud
similar en pérdida esperada — consistente con la experiencia real de
aseguradoras de salud durante COVID-19. Esto refuerza la importancia
de modelar correlación explícitamente en capital económico, en vez de
asumir independencia entre asegurados.

### Limitación
rho=0.15 y los multiplicadores de severidad son supuestos de escenario
(juicio experto), no estimaciones de los datos — MEPS no permite
estimar correlación sistémica real entre personas. Sensibilidad a
estos parámetros no explorada en esta iteración; sería un siguiente
paso natural (análisis de sensibilidad variando rho).

___________________________________________________________________

## Nota técnica — Dashboard Streamlit (dashboard/)

Se detectó incompatibilidad sistemática entre gráficos de barras de 
Plotly (go.Bar y px.bar) y el entorno de renderizado local (versión de 
Streamlit/navegador) — las barras no se renderizaban correctamente 
(altura incorrecta, apilado inesperado, o ejes mal interpretados) en 
múltiples páginas (Segmentos, Stress Test). Elementos de línea de 
Plotly (add_hline) sí funcionaban con normalidad.

Solución aplicada: se reemplazaron los gráficos de barra por 
st.bar_chart (componente nativo de Streamlit), que renderiza 
correctamente en este entorno. Streamlit instalado es una versión 
anterior a la que soporta el parámetro stack=False de st.bar_chart 
— para comparar múltiples métricas sin apilar, se usan gráficos 
separados en columnas en vez de un solo gráfico agrupado.

________________________________________________________________________

## Tests automatizados (test/)

Suite de pytest cubriendo las regresiones ya detectadas durante el
desarrollo:
- ETL: columnas esperadas, ausencia de NaN en conteos (regresión del
  bug de meps_2023_clean.csv desactualizado), completitud de años,
  no negatividad de gasto, validez de flags binarios, sincronización
  entre longitudinal y corte 2023, ausencia de duplicados persona+año.
- Modelos: severidad siempre positiva y en rango razonable,
  probabilidad de gasto en [0,1], fallo explícito ante features
  faltantes, coherencia de prima_pura (regresión del bug de doble
  conteo frecuencia×severidad_anual, ~120x de sobreestimación
  detectado durante el desarrollo de pricing_engine.py).

____________________________________________________________________________

## Tests automatizados (tests/)

Suite de pytest: 28 tests, cubriendo ETL (7), modelos persistidos (5)
y API FastAPI (16). Incluye regresiones de los bugs ya documentados
(NaN espurios en n_diagnosticos_unicos, doble conteo en prima_pura) y
validación de hallazgos de negocio (credibilidad coherente por
tipo_cobertura, mayor capital económico proporcional en escenario de
pandemia vs catástrofe regional). 28/28 passing, ~14s de ejecución
total. Comando: `pytest -v` desde la raíz del proyecto.

Nota técnica menor: warning de deprecación de Pydantic (Config
basado en clase → ConfigDict) en PersonaInput — no afecta
funcionamiento, pendiente de migración en iteración futura.

____________________________________________________________________________

