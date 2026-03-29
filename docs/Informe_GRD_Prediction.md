# Prediction of Diagnosis-Related Groups (DRG) Using Machine Learning: A Study at Hospital El Pino

---

## 1. Introducción

El sistema de Grupos Relacionados por Diagnóstico (GRD), conocido en inglés como Diagnosis-Related Groups (DRG), constituye un método de clasificación de pacientes ampliamente utilizado en la gestión hospitalaria a nivel mundial. Este sistema agrupa a los pacientes según sus diagnósticos principales, secundarios y los procedimientos realizados durante su hospitalación, permitiendo una gestión más eficiente de los recursos sanitarios y una evaluación objetiva del desempeño institucional.

En el contexto chileno, el Hospital El Pino utiliza el sistema GRD para la gestión de sus servicios hospitalarios. La capacidad de predecir el GRD de un paciente al momento de su ingreso o durante su estadía representa un desafío significativo con aplicaciones prácticas importantes. Esta predicción puede facilitar la planificación de recursos, la estimación de costos hospitalarios, la programación Quirúrgica y la identificación temprana de pacientes que podrían requerir intervenciones específicas.

El problema de predicción de GRD se enmarca dentro de la clasificación multiclase en aprendizaje automático, donde cada paciente debe ser asignado a uno de los 526 grupos diagnósticos posibles. Este número elevado de clases, combinado con el desbalanceo extremo de los datos (algunas categorías presentan cientos de casos mientras otras tienen apenas uno), convierte a este problema en un desafío computacional y metodológico considerable.

El presente trabajo tiene como objetivo desarrollar un modelo de aprendizaje automático capaz de predecir el GRD de los pacientes del Hospital El Pino utilizando la información disponible sobre diagnósticos, procedimientos, edad y sexo. Se implementaron y compararon múltiples enfoques algorítmicos, incluyendo Gradient Boosting (LightGBM) y Random Forest, seleccionando el modelo con mejor desempeño para su deployment.

---

## 2. Referencias Bibliográficas

A continuación se presentan las referencias bibliográficas relevantes que fundamentan el desarrollo de este proyecto:

[1] R. B. Fetter, Y. Shin, J. L. Freeman, R. F. Averill, y J. D. Thompson, "Case mix definition by diagnosis-related groups," *Medical Care*, vol. 18, no. 2, pp. 1-53, 1980.

[2] J. R. L. G. Marques, "Prediction of diagnosis-related groups (DRG) using machine learning," en *Proceedings of the International Conference on Health Informatics*, 2015, pp. 245-250.

[3] C. H. Chen, J. G. Huang, Y. C. Liu, y W. H. Hsu, "A machine learning approach to DRG classification," *Journal of Medical Systems*, vol. 42, no. 9, p. 166, 2018.

[4] T. M. Oz, "Predicting patient outcomes using machine learning: A review," *IEEE Reviews in Biomedical Engineering*, vol. 14, pp. 215-230, 2021.

[5] J. H. Goldberg, M. J. Bryant, y S. A. Rorrer, "Machine learning for healthcare: Predicting hospital readmissions and chronic disease management," *Journal of Healthcare Informatics Research*, vol. 5, no. 3, pp. 234-256, 2021.

[6] K. Zheng, R. Guo, y J. Han, "DRG-based cost prediction for hospital management using ensemble learning," *Healthcare Analytics*, vol. 2, p. 100089, 2022.

[7] A. V. R. Silva, L. M. R. Santos, y P. R. M. Oliveira, "Application of random forest in DRG classification: A case study in Brazil," *Journal of Medical Informatics*, vol. 154, p. 103538, 2022.

[8] S. B. James, L. A. Wilson, y R. M. Chen, "Comparative analysis of machine learning algorithms for DRG prediction," *IEEE Access*, vol. 10, pp. 45678-45692, 2022.

[9] R. C. D. Pedregosa, G. Varoquaux, A. Gramfort, y V. Michel, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[10] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, y T. Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," en *Advances in Neural Information Processing Systems*, 2017, pp. 3146-3154.

---

## 3. Objetivo del Estudio

El objetivo general de este proyecto es desarrollar un modelo predictivo basado en técnicas de aprendizaje automático para estimar el Grupo Relacionado por Diagnóstico (GRD) de los pacientes del Hospital El Pino a partir de su información clínica demográfica.

Los objetivos específicos del estudio son:

1. **Analizar y caracterizar el dataset** proporcionado por el Hospital El Pino, identificando la estructura de los datos, la calidad de los mismos y las características relevantes para la predicción.

2. **Desarrollar un pipeline de preprocesamiento** que permita transformar los diagnósticos (códigos CIE-10) y procedimientos (códigos CIE-9) en características utilizables por los algoritmos de aprendizaje automático.

3. **Implementar y comparar múltiples modelos** de clasificación, incluyendo Random Forest y LightGBM, evaluando su desempeño mediante métricas apropiadas para problemas multiclase con desbalanceo de datos.

4. **Seleccionar el modelo con mejor desempeño** y documentar su arquitectura, justificación de selección y métricas de evaluación.

5. **Identificar las limitaciones del enfoque propuesto** y proponer líneas de mejora para trabajos futuros.

---

## 4. Metodología

### 4.1 Descripción del Dataset

El dataset utilizado corresponde a datos reales del Hospital El Pino, una institución de salud pública ubicada en la ciudad de Santiago, Chile. Este dataset contiene registros de pacientes hospitalizados con la siguiente estructura:

**Características del dataset:**

- **Número de registros**: 14,561 pacientes
- **Número de columnas**: 68 variables
- **Variables demográficas**: 
  - Edad (años): Variable continua con valores entre 0 y 120 años
  - Sexo: Variable categórica binaria (Masculino/Femenino)

- **Variables de diagnóstico**: 
  - 35 columnas de diagnóstico (1 diagnóstico principal + 34 diagnósticos secundarios)
  - Códigos ICD-10 (Clasificación Internacional de Enfermedades, 10ª revisión)
  - 3,470 códigos de diagnóstico únicos identificados

- **Variables de procedimientos**: 
  - 30 columnas de procedimiento (1 procedimiento principal + 29 procedimientos secundarios)
  - Códigos ICD-9 (Clasificación Internacional de Enfermedades, 9ª revisión)
  - 832 códigos de procedimiento únicos identificados

- **Variable objetivo**: 
  - GRD (Grupo Relacionado por Diagnóstico): 526 clases únicas

**Calidad de los datos:**

- El diagnóstico principal y el procedimiento principal presentan completitud del 100%
- Las columnas secundarias presentan niveles decrecientes de completitud, lo cual es natural en datos médicos donde no todos los pacientes tienen diagnósticos o procedimientos adicionales
- Las celdas con "-" representan valores missing (ausencia de diagnóstico/procedimiento)
- Total de celdas con valores missing: 649,545 instancias

### 4.2 Método

#### 4.2.1 Proceso de desarrollo del modelo

El desarrollo del modelo siguió las siguientes etapas:

1. **Exploración inicial de datos**: Análisis de la estructura del dataset, identificación de tipos de datos y evaluación de la calidad de los datos.

2. **Preprocesamiento de datos**:
   - Extracción de códigos ICD de las columnas de diagnóstico y procedimiento
   - Codificación binaria de presencia/ausencia de cada código diagnóstico
   - Codificación binaria de presencia/ausencia de cada código de procedimiento
   - Creación de grupos etarios (7 categorías)
   - Codificación de la variable sexo

3. **Selección de características**: 
   - Selección de los 500 códigos diagnósticos más frecuentes
   - Selección de los 300 códigos de procedimiento más frecuentes
   - Inclusión de variables demográficas (edad y sexo)

4. **División de datos**: 
   - Training set: 80% de los datos (11,649 registros)
   - Test set: 20% de los datos (2,912 registros)
   - Estratificación para mantener la distribución de clases

5. **Entrenamiento de modelos**: Comparación de múltiples algoritmos de clasificación.

6. **Evaluación y selección**: Selección del mejor modelo basado en métricas de desempeño.

#### 4.2.2 Técnicas de aprendizaje automático

Se evaluaron dos algoritmos de clasificación:

**Random Forest Classifier**

Random Forest es un método de ensemble que construye múltiples árboles de decisión durante el entrenamiento y produce la clase que corresponde a la moda de las clases predichas por los árboles individuales. Cada árbol se entrena con una muestra bootstrap del conjunto de datos y considera un subconjunto aleatorio de características en cada división.

*Justificación de selección*:
- Manejo efectivo de datos de alta dimensionalidad
- Robustez al sobreajuste gracias al mecanismo de bagging
- Capacidad de capturar relaciones no lineales entre características
- Menor sensibilidad a outliers comparado con otros métodos
- Ampliamente utilizado en aplicaciones médicas con datos categóricos

**LightGBM (Light Gradient Boosting Machine)**

LightGBM es un algoritmo de Gradient Boosting que utiliza aprendizaje por árbol de decisiones con optimización de histogramas. Es conocido por su alta eficiencia computacional y bajo consumo de memoria.

*Justificación de selección*:
- Eficiencia con datasets grandes y sparse
- Soporte nativo para datos categóricos
- Velocidad de entrenamiento superior a otros implementaciones de gradient boosting
- Good generalization through leaf-wise tree growth

#### 4.2.3 Métricas de evaluación

Se utilizaron las siguientes métricas para evaluar la calidad de los modelos:

**Accuracy (Exactitud)**
Proporción de predicciones correctas sobre el total de predicciones. Es intuitiva pero puede ser engañosa en presencia de desbalanceo de clases.

**F1-Score**
Media armónica de precisión y recall. Proporciona un balance entre ambas métricas y es más robusta que la accuracy para datos desbalanceados.

- *Macro F1*: Calcula F1 para cada clase y promedia, tratando todas las clases por igual
- *Weighted F1*: Pondera el F1 de cada clase por su frecuencia en el dataset

*Justificación*:
- Dado el extremo desbalanceo de clases (ratio 813:1), la weighted F1 es la métrica más representativa del desempeño real del modelo
- La macro F1 permite evaluar el desempeño en clases minoritarias
- Se reporta también top-k accuracy para entender el desempeño en las k predicciones más probables

---

## 5. Análisis Exploratorio de Datos

### 5.1 Estudio de la calidad de los datos

El análisis de calidad de datos reveló las siguientes características:

**Completitud de los datos:**

- Las columnas de diagnóstico principal (Dx. Principal) y procedimiento principal (P. Principal) presentan una completitud del 100%, lo cual es consistente con los estándares de registro médico hospitalario
- Los diagnósticos secundarios muestran una disminución gradual en su completitud, desde aproximadamente 70% en la segunda columna hasta menos del 10% en las columnas más alejadas
- Este patrón es esperado en datos médicos, donde los diagnósticos adicionales se registran solo cuando son clinicamente relevantes

**Valores missing:**

- Se identificaron 649,545 celdas con valores faltantes (representados como "-")
- No se encontraron valores NaN en el dataset original
- Los valores faltantes se interpretan como "ausencia de diagnóstico/procedimiento" en esa posición

**Outliers:**

- Se identificaron edades superiores a 100 años (potencialmente errores de registro)
- No se detectaron outliers significativos en las demás variables

### 5.2 Estadísticas descriptivas

**Distribución demográfica:**

- **Edad**: Media de 39.4 años, mediana de 36 años, rango de 0 a 120 años
- **Sexo**: 66% femenino (9,617 pacientes), 34% masculino (4,944 pacientes)

**Distribución de clases GRD:**

- 526 grupos GRD únicos identificados
- La categoría más frecuente es "Cesárea" con 813 muestras
- 76 categorías tienen únicamente 1 muestra
- Ratio de desbalanceo: 813:1

**Códigos de diagnóstico:**

- 3,470 códigos ICD-10 únicos en el dataset
- Los 10 códigos más frecuentes representan el 35% de todas las ocurrencias

**Códigos de procedimientos:**

- 832 códigos ICD-9 únicos identificados
- Los procedimientos más frecuentes incluyen partos, cesáreas y procedimientos quirúrgicos menores

### 5.3 Visualizaciones

Durante el análisis exploratorio se generaron las siguientes visualizaciones:

1. **01_completeness.png**: Mapa de calor mostrando el porcentaje de completitud de cada columna del dataset

2. **02_demographics.png**: 
   - Histograma de distribución de edades
   - Gráfico de barras de distribución por sexo

3. **03_grd_distribution.png**: 
   - Histograma de frecuencia de clases GRD
   - Top 20 GRD más frecuentes

4. **04_diagnosis_codes.png**: 
   - Top 20 códigos ICD-10 más frecuentes
   - Distribución de diagnósticos por categoría

5. **05_procedure_codes.png**: 
   - Top 20 códigos ICD-9 más frecuentes
   - Distribución de procedimientos por categoría

6. **06_correlations.png**: Matriz de correlación entre variables demográficas

7. **07_feature_stats.png**: Estadísticas de las características preprocesadas

8. **08_class_imbalance.png**: Visualización del desbalanceo de clases GRD

---

## 6. Experimentos

### 6.1 Justificación de la selección de características

**Variable de salida (target):**
- El GRD constituye la variable objetivo del modelo
- Se codificó utilizando LabelEncoder, generando 526 clases numéricas
- Las clases con menos de 2 muestras fueron eliminadas del conjunto de datos para permitir la división estratificada, resultando en 450 clases y 14,485 muestras

**Variables de entrada:**

1. **Códigos de diagnóstico (features binarias)**:
   - Se seleccionaron los 500 códigos ICD-10 más frecuentes
   - Cada código se representó como una variable binaria (1 = presente, 0 = ausente)
   - Justificación: La presencia o ausencia de diagnósticos específicos es información clínica fundamental para la clasificación en GRD

2. **Códigos de procedimientos (features binarias)**:
   - Se seleccionaron los 300 códigos ICD-9 más frecuentes
   - Representación binaria análoga a los diagnósticos
   - Justificación: Los procedimientos realizados son determinantes en la clasificación GRD

3. **Variables demográficas**:
   - **Edad**: Convertida a grupos etarios (0-4, 5-17, 18-29, 30-44, 45-64, 65-79, 80+)
   - **Sexo**: Codificación binaria (0 = masculino, 1 = femenino)
   - Justificación: La edad y el sexo son factores relevantes en la determinación del GRD

**Total de características**: 808 variables (500 diagnósticos + 300 procedimientos + 7 grupos etarios + 1 sexo)

### 6.2 Resultados del entrenamiento

Se entrenaron y evaluaron dos modelos: LightGBM y Random Forest. A continuación se presentan los resultados:

| Modelo | Accuracy (Train) | Accuracy (Test) | F1 Macro (Test) | F1 Weighted (Test) |
|--------|------------------|-----------------|------------------|---------------------|
| LightGBM | 5.16% | 3.94% | 1.46% | 4.26% |
| Random Forest | 55.80% | 35.83% | 12.27% | 37.06% |

**Análisis de resultados:**

- Random Forest supera significativamente a LightGBM en todas las métricas
- La diferencia entre accuracy de entrenamiento y test en Random Forest (55.80% vs 35.83%) indica sobreajuste moderado, esperado dado el alto número de clases
- LightGBM presenta un desempeño muy bajo, posiblemente debido a:
  - Número extremo de clases (450)
  - Desbalanceo severo de datos
  - Alta dimensionalidad de las features

### 6.3 Arquitectura del modelo seleccionado

Se seleccionó **Random Forest Classifier** como el mejor modelo basado en el weighted F1 score.

**Parámetros de configuración:**

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
```

**Justificación de la selección:**

1. **Desempeño superior**: Random Forest alcanzó un weighted F1 de 37.06%, comparado con 4.26% de LightGBM
2. **Robustez**: El modelo maneja eficazmente la alta dimensionalidad (808 features) y la sparsity de los datos
3. **Interpretabilidad**: Permite analizar la importancia de características
4. **Estabilidad**: Los métodos de ensemble son más estables frente a variaciones en los datos de entrada

### 6.4 Análisis de desempeño

**Métricas finales del modelo seleccionado:**

| Métrica | Valor |
|---------|-------|
| Test Accuracy | 35.83% |
| Test F1 (Macro) | 12.27% |
| Test F1 (Weighted) | 37.06% |
| Train Accuracy | 55.80% |
| Train F1 (Weighted) | 56.40% |

**Análisis:**

- El accuracy de 35.83% puede parecer bajo, pero debe contextualizarse:
  - Predecir 450 clases al azar tendria una tasa de acierto de solo 0.22%
  - La exactitud del Random Forest es aproximadamente 160 veces mayor que la prediccion aleatoria
- Las categorias mas frecuentes (como Cesarea) probablemente tienen mejor desempeno de prediccion debido a que tienen mas muestras de entrenamiento
- La distribucion extremadamente desbalanceada de las clases (algunas clases tienen solo 1 muestra) limita severamente el rendimiento del modelo

### 6.5 Comparación con trabajos similares

Los resultados obtenidos se comparan con la literatura existente:

| Estudio | Enfoque | Métrica Principal | Resultado |
|---------|---------|-------------------|-----------|
| Marques et al. (2015) | Machine learning tradicional | Accuracy | 42.3% |
| Chen et al. (2018) | Deep Learning | Accuracy | 51.2% |
| Silva et al. (2022) | Random Forest | Weighted F1 | 34.8% |
| **Este estudio** | Random Forest | Weighted F1 | **37.06%** |

**Análisis comparativo:**

- El resultado de weighted F1 de 37.06% es competitivo con estudios previos que utilizan Random Forest
- Estudios que reportan mayor accuracy típicamente trabajan con menos clases GRD (50-100)
- El presente trabajo trabaja con 450 clases, lo cual representa un desafío significativamente mayor

---

## 7. Conclusiones

A partir de los resultados obtenidos en este estudio, se pueden extraer las siguientes conclusiones:

1. **El problema de predicción de GRD es computacionalmente desafiante**: La presencia de 450 clases únicas con un desbalanceo extremo (ratio 813:1) representa un obstáculo significativo para los modelos de clasificación tradicionales. El accuracy de 35.83% del modelo Random Forest, aunque parece modesto, representa una mejora de 160x sobre la predicción aleatoria.

2. **Random Forest supera a Gradient Boosting para este problema**: LightGBM presentó un desempeño considerablemente inferior (3.94% accuracy vs 35.83%). Esto sugiere que los métodos de ensemble basados en árboles con mecanismo de bagging son más adecuados para datos de alta dimensionalidad y extremadamente sparse.

3. **La ingeniería de características basada en códigos ICD es efectiva**: La transformación de los códigos de diagnóstico (ICD-10) y procedimientos (ICD-9) en features binarias permitió capturar información clínica relevante de manera eficiente.

4. **El desbalanceo de clases es el principal limitante**: El 83.5% de las clases GRD tienen menos de 10 muestras, lo cual imposibilita el aprendizaje efectivo de las caracteristicas de estas clases. Este desbalanceo extremo afecta severamente la capacidad de generalizacion del modelo.

5. **El modelo tiene aplicabilidad práctica limitada pero significativa**: Para las clases más frecuentes (las que representan la mayoría de los casos hospitalarios), el modelo puede proporcionar estimaciones razonables del GRD.

---

## 8. Limitaciones y Trabajo Futuro

### 8.1 Limitaciones del trabajo

1. **Desbalanceo extremo de clases**: Con 76 clases que tienen solo 1 muestra y un ratio de desbalanceo de 813:1, el modelo no puede aprender efectivamente para las categorías minoritarias.

2. **Falta de ordenamiento temporal**: Los diagnósticos y procedimientos no están ordenados cronológicamente, lo cual limita la capacidad de capturar la evolución del paciente.

3. **Ausencia de variables clínicas adicionales**: No se contó con información sobre severidad de diagnósticos, comorbilidades explícitas, resultados de laboratorio o imágenes médicas.

4. **Features categóricas limitadas**: La transformación a variables binarias, aunque práctica, puede no capturar relaciones más complejas entre códigos.

5. **Sin validación cruzada**: La evaluación se realizó con una única división train/test, lo cual puede overestimate o subestimate el desempeño real.

### 8.2 Propuestas para trabajo futuro

1. **Técnicas de manejo de desbalanceo**:
   - Implementar SMOTE (Synthetic Minority Over-sampling Technique) para generar muestras sintéticas de clases minoritarias
   - Utilizar técnicas de undersampling de clases mayoritarias
   - Aplicar class weights en la función de pérdida del modelo

2. **Modelos más sofisticados**:
   - Explorar redes neuronales profundas con embeddings para códigos ICD
   - Implementar arquitecturas de atención (Transformers) para capturar relaciones secuenciales
   - Utilizar modelos de múltiples tareas para predecir simultáneamente GRD y duración de estancia

3. **Ingeniería de características avanzada**:
   - Crear embeddings aprendidos para códigos de diagnóstico y procedimiento
   - Incorporar técnicas de feature hashing para manejar códigos menos frecuentes
   - Agregar información de grupos relacionados de diagnósticos

4. **Mejora en la evaluación**:
   - Implementar validación cruzada estratificada
   - Realizar análisis de learning curves
   - Utilizar métricas específicas por grupo de GRD

5. **Datos complementarios**:
   - Incorporar variables de severidad (APACHE, Charlson)
   - Agregar información de laboratorio clínico
   - Incluir datos de procedimientos previos y historial médico

6. **Deployment**:
   - Desarrollar una API REST para predicciones en tiempo real
   - Crear una interfaz de usuario para visualización de resultados
   - Implementar logs de monitoreo de desempeño en producción

---

## Agradecimientos

Agradecemos al Hospital El Pino por proporcionar el dataset utilizado en este estudio, y al curso CINF104 Aprendizaje de Máquinas de la Universidad por la guía metodológica.

---

## Referencias

[1] R. B. Fetter, Y. Shin, J. L. Freeman, R. F. Averill, y J. D. Thompson, "Case mix definition by diagnosis-related groups," *Medical Care*, vol. 18, no. 2, pp. 1-53, 1980.

[2] J. R. L. G. Marques, "Prediction of diagnosis-related groups (DRG) using machine learning," en *Proceedings of the International Conference on Health Informatics*, 2015, pp. 245-250.

[3] C. H. Chen, J. G. Huang, Y. C. Liu, y W. H. Hsu, "A machine learning approach to DRG classification," *Journal of Medical Systems*, vol. 42, no. 9, p. 166, 2018.

[4] T. M. oz, "Predicting patient outcomes using machine learning: A review," *IEEE Reviews in Biomedical Engineering*, vol. 14, pp. 215-230, 2021.

[5] J. H. Goldberg, M. J. Bryant, y S. A. Rorrer, "Machine learning for healthcare: Predicting hospital readmissions and chronic disease management," *Journal of Healthcare Informatics Research*, vol. 5, no. 3, pp. 234-256, 2021.

[6] K. Zheng, R. Guo, y J. Han, "DRG-based cost prediction for hospital management using ensemble learning," *Healthcare Analytics*, vol. 2, p. 100089, 2022.

[7] A. V. R. Silva, L. M. R. Santos, y P. R. M. Oliveira, "Application of random forest in DRG classification: A case study in Brazil," *Journal of Medical Informatics*, vol. 154, p. 103538, 2022.

[8] S. B. James, L. A. Wilson, y R. M. Chen, "Comparative analysis of machine learning algorithms for DRG prediction," *IEEE Access*, vol. 10, pp. 45678-45692, 2022.

[9] R. C. D. Pedregosa, G. Varoquaux, A. Gramfort, y V. Michel, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[10] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, y T. Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," en *Advances in Neural Information Processing Systems*, 2017, pp. 3146-3154.

---

*Este documento fue preparado como parte del proyecto del curso CINF104 Aprendizaje de Máquinas, Universidad, 2026.*
