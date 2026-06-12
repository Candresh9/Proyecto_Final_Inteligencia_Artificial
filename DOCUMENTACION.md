# 📘 Documentación Técnica del Proyecto (Versión Nativa en Python)

Esta documentación detalla la arquitectura, las funciones y los fundamentos matemáticos de la aplicación interactiva **`app.py`**, desarrollada en **Python** utilizando **Streamlit**, **Pandas**, **NumPy** y **Plotly** como plataforma educativa para el estudio y comparación de modelos de Regresión Lineal y Regresión Logística.

---

## 1. Arquitectura del Software y Estado Global

La aplicación está diseñada de manera modular y estructurada bajo un único archivo ejecutable (`app.py`), el cual expone dos módulos lógicos principales a través de un selector de navegación en la barra lateral. 

Debido al flujo reactivo de Streamlit (donde la aplicación se re-ejecuta por completo ante cualquier interacción del usuario), se utiliza **`st.session_state`** para mantener la persistencia de los datos, coeficientes y configuraciones del modelo:

### 📊 Variables del Módulo 1 (Simulador 2D)
*   `st.session_state.df`: DataFrame de Pandas que almacena los puntos del plano cartesiano (columnas: `X`, `Y`, `Clase`).
*   `st.session_state.model_m` / `st.session_state.model_b`: Pendiente ($m$) e intercepto ($b$) del modelo de regresión actual.
*   `st.session_state.history`: Lista de valores del costo $MSE$ a lo largo de las iteraciones.
*   `st.session_state.trained`: Indicador booleano de si el modelo ha sido entrenado.

### 🧬 Variables del Módulo 2 (Regresión Logística Multivariable)
*   `st.session_state.multi_df`: DataFrame con el conjunto de datos multivariable cargado.
*   `st.session_state.multi_train_df` / `st.session_state.multi_test_df`: Partición del conjunto de datos para entrenamiento y prueba.
*   `st.session_state.multi_target` / `st.session_state.multi_features`: Nombre de la variable objetivo y lista de variables predictoras.
*   `st.session_state.multi_feature_stats`: Medias y desviaciones estándar del set de entrenamiento para la normalización Z-score.
*   `st.session_state.multi_weights` / `st.session_state.multi_bias`: Vector de pesos y sesgo del clasificador logístico.
*   `st.session_state.multi_cost_history`: Historial de pérdida Log-Loss por época de entrenamiento.
*   `st.session_state.multi_trained`: Indicador booleano de si el clasificador logístico está listo.
*   `st.session_state.multi_threshold`: Umbral de decisión ($T$) para clasificar probabilidades en etiquetas discretas ($0$ o $1$).

---

## 2. Detalle de Funciones y Algoritmos de Machine Learning

### 📊 Funciones del Módulo 1 (Simulador 2D)

#### 1. `change_data_source()`
*   **Propósito:** Cambiar dinámicamente el conjunto de puntos en el simulador según la selección del usuario.
*   **Funcionamiento:**
    *   Si se selecciona `"Traslapado (Preset)"`, carga una matriz predefinida donde las dos clases se mezclan en el medio.
    *   Si se selecciona `"Generar Aleatorio"`, inicializa una semilla aleatoria (`np.random.seed(42)`), genera $15$ muestras uniformes en $X$, calcula etiquetas binarias aplicando ruido normal en un umbral de $0.5$, asigna coordenadas $Y$ congruentes en rangos diferenciados para cada clase y carga el conjunto en la sesión.
    *   Establece el indicador de entrenamiento en `False` y limpia el historial de pérdidas.

#### 2. `fit_ols(df)`
*   **Propósito:** Calcular la pendiente ($m$) y el intercepto ($b$) óptimos de la recta de regresión de forma directa.
*   **Algoritmo:** Implementa la solución analítica de **Mínimos Cuadrados Ordinarios (OLS)**.
*   **Ecuación Matemática:**
    $$m = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{N} (x_i - \bar{x})^2} \qquad b = \bar{y} - m\bar{x}$$
    Si la varianza del denominador es $0.0$ (todos los puntos en la misma línea vertical), coloca por defecto la pendiente en $0.0$ y el intercepto en la media de Y ($\bar{y}$).

#### 3. `calculate_metrics(df, m, b, threshold)`
*   **Propósito:** Calcular las métricas de error del ajuste lineal continuo y los indicadores de desempeño de la clasificación discreta.
*   **Parámetros:** `df` (DataFrame de puntos), `m` (pendiente), `b` (intercepto) y `threshold` (umbral de clasificación).
*   **Métricas de Regresión:**
    *   **Error Cuadrático Medio (MSE):** Mide el promedio de las distancias verticales al cuadrado entre la recta de regresión y los puntos reales:
        $$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - (m \cdot x_i + b))^2$$
    *   **Coeficiente de Determinación ($R^2$):** Proporción de la varianza total de la variable $Y$ explicada por el modelo de regresión:
        $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
*   **Métricas de Clasificación:** Clasifica cada punto según si la predicción de la recta $\hat{y}_i \ge \text{threshold}$ (Clase 1) o no (Clase 0). Construye la matriz de confusión (TN, FP, FN, TP) y devuelve los indicadores estándar:
    *   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
    *   $$\text{Precision} = \frac{TP}{TP + FP}$$
    *   $$\text{Recall} = \frac{TP}{TP + FN}$$
    *   $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### 🧬 Funciones del Módulo 2 (Regresión Logística Multivariable)

#### 1. `sigmoid(z)`
*   **Propósito:** Función de activación que mapea cualquier valor real al intervalo $(0, 1)$, permitiendo interpretar la salida lineal del modelo como una probabilidad.
*   **Ecuación Matemática:**
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
    *Nota: Utiliza `np.clip` en el rango $[-500, 500]$ para prevenir desbordamientos numéricos por valores extremos.*

#### 2. `train_logistic_regression(X_train, y_train, learning_rate, epochs)`
*   **Propósito:** Optimizar iterativamente los pesos $\mathbf{w}$ y el sesgo $b$ mediante **Descenso de Gradiente** de lote completo (Batch Gradient Descent).
*   **Algoritmo:**
    1.  Inicializa los pesos $\mathbf{w}$ en cero y el sesgo $b$ en $0.0$.
    2.  En cada época, calcula la predicción de probabilidad para cada muestra:
        $$\hat{\mathbf{y}} = \sigma(\mathbf{X} \mathbf{w} + b)$$
    3.  Calcula el costo de **Entropía Cruzada Binaria (Log-Loss)** (con acotamiento a $[10^{-15}, 1 - 10^{-15}]$ para evitar indeterminaciones matemáticas):
        $$J(\mathbf{w}, b) = -\frac{1}{M} \sum_{i=1}^{M} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
    4.  Calcula los gradientes parciales del costo respecto a los pesos y al sesgo:
        $$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{M} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
        $$\frac{\partial J}{\partial b} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)$$
    5.  Actualiza los parámetros en dirección opuesta al gradiente multiplicado por la tasa de aprendizaje ($\alpha$):
        $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}} \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
    6.  Devuelve los pesos finales, el sesgo y la lista histórica de costos.

#### 3. `predict_probabilities(X, weights, bias)`
*   **Propósito:** Estimar la probabilidad de pertenencia a la Clase 1 de un conjunto de datos `X`.
*   **Algoritmo:** Multiplica las variables predictoras por sus respectivos pesos, añade el sesgo y aplica la función sigmoide:
    $$\text{Probabilidad} = \sigma(\mathbf{X} \mathbf{w} + b)$$

#### 4. `calculate_roc_auc(y_true, probs)`
*   **Propósito:** Evaluar la calidad de discriminación del clasificador binario sobre el set de prueba.
*   **Algoritmo:**
    *   Itera el umbral de decisión $T$ desde $1.0$ hasta $0.0$ en $101$ puntos.
    *   Para cada umbral, calcula la **Tasa de Verdaderos Positivos (TPR)** (Sensibilidad) y la **Tasa de Falsos Positivos (FPR)** ($1 - \text{Especificidad}$):
        $$\text{TPR} = \frac{TP}{TP + FN} \qquad \text{FPR} = \frac{FP}{FP + TN}$$
    *   Calcula numéricamente el **Área bajo la curva ROC (AUC)** integrando los trapecios formados por los puntos ordenados de FPR y TPR:
        $$\text{AUC} = \sum_{i=1}^{K} \frac{\text{TPR}_i + \text{TPR}_{i-1}}{2} \cdot (\text{FPR}_i - \text{FPR}_{i-1})$$
    *   Devuelve los arreglos de FPR, TPR y el valor flotante del AUC.

---

## 3. Visualizaciones Científicas en Plotly

La aplicación en Python reemplaza los Canvas 2D originales de JavaScript y los gráficos estáticos de Matplotlib por visualizaciones dinámicas interactivas usando la librería **Plotly**:

1.  **Plano del Simulador 2D:** Dibuja los puntos coloreados según su etiqueta de clase, traza la recta de regresión lineal ajustada ($y = mx + b$), traza la frontera de decisión ($x_{\text{frontera}} = \frac{T - b}{m}$) y añade rectángulos translúcidos en el fondo para ilustrar las regiones asignadas a cada clase.
2.  **Visualización Multivariable 2D:** Permite al usuario seleccionar dos características numéricas cualesquiera del dataset y grafica un diagrama de dispersión de los puntos coloreados por su clase real. Si el modelo logístico ya fue entrenado, proyecta la frontera de decisión multivariable sobre esas dos características evaluadas en su media muestral.
3.  **Curva ROC de Prueba:** Grafica la Tasa de Falsos Positivos contra la Tasa de Verdaderos Positivos e incluye la línea de azar diagonal (recta discontinua desde $(0,0)$ hasta $(1,1)$) para evaluar de forma directa el poder predictivo del clasificador.

---

## 4. Requisitos e Instrucciones de Ejecución

Para iniciar la aplicación educativa en tu servidor local:

1.  **Instalar dependencias declaradas:**
    Asegúrate de contar con Python 3.9 o superior y corre en tu terminal:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Iniciar la aplicación con Streamlit:**
    Ejecuta el servidor web local:
    ```bash
    streamlit run app.py
    ```

3.  **Visualización en Navegador:**
    Abre tu navegador de internet e ingresa a la dirección por defecto indicada en la terminal:
    `http://localhost:8501`
