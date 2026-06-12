# 🐍 Guía de Estudio: Explicación del Código Python (`app.py`)
Este documento explica de forma exclusiva y detallada cómo funciona el archivo **`app.py`**, detallando su estructura, el propósito de cada una de sus funciones y la matemática que ejecuta bajo el capó.

---

## 1. Librerías Utilizadas
El código comienza importando las siguientes herramientas clave para la Inteligencia Artificial y la interactividad:
*   `streamlit (st)`: Orquesta toda la interfaz gráfica, el estado de la aplicación y la interactividad web.
*   `pandas (pd)`: Estructura los datos de los puntos de entrenamiento en tablas (`DataFrames`).
*   `numpy (np)`: Realiza los cálculos matemáticos avanzados y las operaciones vectorizadas de forma rápida.
*   `matplotlib.pyplot (plt)`: Genera gráficos (aunque en esta versión moderna el graficador principal es Plotly).
*   `plotly.graph_objects (go)`: Genera el plano cartesiano interactivo 2D permitiendo hacer zoom, ver coordenadas al pasar el cursor y personalizar colores.
*   `time`: Introduce retardos temporales para poder visualizar la animación del Descenso de Gradiente paso a paso.

---

## 2. Gestión del Estado de la Aplicación (`st.session_state`)
Streamlit es un framework reactivo: cada vez que el usuario mueve un control deslizante o edita una celda, todo el script `app.py` se vuelve a ejecutar de arriba a abajo. Para evitar perder los datos calculados o los puntos creados en la ejecución anterior, el código utiliza variables persistentes llamadas **`st.session_state`**:

*   `st.session_state.df`: El conjunto de datos actual con columnas `X`, `Y` y `Clase`.
*   `st.session_state.model_m` / `st.session_state.model_b`: La pendiente ($m$) y el intercepto ($b$) del modelo de regresión actual ($y = mx + b$).
*   `st.session_state.history`: Guarda el historial de valores del Error Cuadrático Medio ($MSE$) por cada época durante el Descenso de Gradiente para graficar la curva de aprendizaje.
*   `st.session_state.trained`: Un indicador booleano (`True`/`False`) de si el modelo ya tiene coeficientes entrenados.

---

## 3. Desglose de Funciones de Python

### 1. `change_data_source()`
*   **¿Cuándo se ejecuta?** Cuando el usuario cambia la fuente de datos en el menú desplegable del sidebar.
*   **¿Qué hace?**
    *   Si el usuario elige `"Traslapado (Preset)"`, carga un conjunto de datos donde las clases se cruzan en el medio.
    *   Si elige `"Generar Aleatorio"`, crea matemáticamente $15$ puntos artificiales. Genera coordenadas $X$ al azar entre $0.1$ y $0.9$. A las clases las define con base en si $X$ (con un poco de ruido normal) es mayor a $0.5$. Luego, a los puntos de Clase 1 les asigna un $Y$ alto ($0.5$ a $0.9$) y a los de Clase 0 un $Y$ bajo ($0.1$ a $0.5$).
    *   Limpia el historial de entrenamiento previo y marca el modelo como no entrenado (`trained = False`).

---

### 2. `fit_ols(df)`
*   **¿Qué hace?** Entrena el modelo utilizando la ecuación matemática exacta de **Mínimos Cuadrados Ordinarios (OLS)**. Es una solución analítica que encuentra la mejor recta de forma instantánea sin dar pasos repetitivos.
*   **Fórmulas Matemáticas implementadas:**
    La pendiente ($m$) se calcula dividiendo la covarianza de $X$ e $Y$ por la varianza de $X$:
    $$m = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{N} (x_i - \bar{x})^2}$$
    El intercepto ($b$) se despeja usando la media de $X$ ($\bar{x}$) y de Y ($\bar{y}$):
    $$b = \bar{y} - m \cdot \bar{x}$$
*   **Control de Errores:** Si todos los puntos tienen la misma coordenada $X$ (lo que daría una división por cero en el denominador), la pendiente $m$ se define por defecto como `0.0` y la recta se coloca horizontalmente a la altura de la media de $Y$.

---

### 3. `calculate_metrics(df, m, b, threshold)`
*   **¿Qué hace?** Evalúa qué tan bueno es el modelo en dos aspectos: en su ajuste continuo (Regresión) y en sus decisiones de etiqueta (Clasificación).
*   **Bajo el Capó:**
    1.  **Predicción Continua:** Calcula la predicción sobre el eje $Y$ para cada punto: $\hat{y}_i = m \cdot x_i + b$.
    2.  **Umbralización (Clasificación):** Convierte el valor continuo en una etiqueta $0$ o $1$ según el umbral ($T$):
        $$\text{Predicción} = \begin{cases} 1 & \text{si } \hat{y}_i \ge T \\ 0 & \text{si } \hat{y}_i < T \end{cases}$$
    3.  **Métricas de Regresión:**
        *   **MSE (Error Cuadrático Medio):** Mide la distancia vertical promedio al cuadrado entre la recta de regresión y los puntos reales:
            $$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
        *   **$R^2$ (Coeficiente de Determinación):** Mide qué porcentaje de la variabilidad de los datos explica la recta. Un valor cercano a $1.0$ representa un ajuste excelente.
    4.  **Métricas de Clasificación:** Compara la predicción binaria con la Clase real y cuenta:
        *   **Verdaderos Positivos (TP):** Clase real = 1, Predicción = 1.
        *   **Verdaderos Negativos (TN):** Clase real = 0, Predicción = 0.
        *   **Falsos Positivos (FP):** Clase real = 0, Predicción = 1.
        *   **Falsos Negativos (FN):** Clase real = 1, Predicción = 0.
    5.  **Cálculo de Proporciones:**
        *   $$\text{Accuracy (Exactitud)} = \frac{TP + TN}{\text{Total de Puntos}}$$
        *   $$\text{Precision (Precisión)} = \frac{TP}{TP + FP}$$
        *   $$\text{Recall (Sensibilidad)} = \frac{TP}{TP + FN}$$
        *   $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 4. Funcionamiento de la Interfaz por Pestañas

### Pestaña 1: Simulador e Interactividad
*   **Tabla Editable (`st.data_editor`):** Permite al usuario editar directamente con decimales las coordenadas y clases de los puntos en pantalla, agregando o eliminando filas. Si el DataFrame cambia, se reinicia el estado y se recarga el sistema automáticamente (`st.rerun()`).
*   **Plano Cartesiano en Plotly:**
    1.  Dibuja las regiones de decisión: Si la pendiente $m$ no es cero, calcula dónde cruza la recta el umbral de decisión: $x_{\text{frontera}} = \frac{T - b}{m}$. Colorea el fondo del plano de verde desde ese punto hacia la derecha (Clase 1) y de rosado hacia la izquierda (Clase 0).
    2.  Dibuja la recta ajustada en color amarillo.
    3.  Dibuja la **Frontera de Decisión** como una línea blanca discontinua en el eje vertical crítico $x_{\text{frontera}}$.
    4.  Dibuja líneas verticales punteadas (**residuales**) que muestran el error existente entre la recta y cada punto del plano.

### Pestaña 2: Entrenamiento en Tiempo Real (Animación del Descenso de Gradiente)
Si el usuario selecciona "Descenso de Gradiente" e inicia la animación:
*   El código inicializa los parámetros en $m = 0.0$ y $b = 0.5$ (recta horizontal).
*   Entra en un bucle `for` por cada época de entrenamiento:
    1.  Calcula la predicción actual y el error de cada muestra: $\text{error}_i = (m \cdot x_i + b) - y_i$.
    2.  Calcula las derivadas parciales de la función de coste (los gradientes):
        $$\text{Gradiente de } m \ (dm) = \frac{2}{N} \sum_{i=1}^{N} (\text{error}_i \cdot x_i)$$
        $$\text{Gradiente de } b \ (db) = \frac{2}{N} \sum_{i=1}^{N} \text{error}_i$$
    3.  Actualiza los coeficientes dando un pequeño paso controlado por la tasa de aprendizaje ($\alpha$):
        $$m \leftarrow m - \alpha \cdot dm$$
        $$b \leftarrow b - \alpha \cdot db$$
    4.  Cada dos épocas, refresca un gráfico interactivo en pantalla que muestra cómo la recta amarilla se va inclinando y acomodando poco a poco sobre los puntos, mientras que una segunda gráfica muestra cómo la curva del coste $MSE$ va descendiendo hasta estabilizarse.

### Pestaña 3: Evaluación de Desempeño
*   Extrae el estado actual del modelo y muestra de forma limpia las métricas calculadas por `calculate_metrics()`.
*   Construye una matriz de confusión en formato de tabla de dos dimensiones para facilitar su lectura en la sustentación.

---

## 5. Preguntas Clave para Estudiar antes de tu Examen

1.  **¿Cómo se calcula la frontera de decisión en este código Python?**
    La frontera es el punto exacto del eje $X$ donde la predicción continua cruza el umbral de clasificación $T$. Se despeja de la ecuación de la recta:
    $$m \cdot x + b = T \implies x_{\text{frontera}} = \frac{T - b}{m}$$

2.  **¿Qué pasa si la pendiente $m$ es 0 en el simulador?**
    Si $m = 0$, la recta es completamente horizontal. Si el intercepto $b$ es mayor o igual al umbral $T$, todo el plano se clasifica como Clase 1. Si es menor, todo el plano se clasifica como Clase 0. No existe un valor de cruce vertical único en el eje $X$.

3.  **¿Por qué un outlier (punto atípico) de la Clase 1 muy alejado en X daña la clasificación en este código?**
    Porque la regresión lineal busca minimizar el error cuadrático ($MSE$). Si hay un punto verde muy alejado con una etiqueta real de $1.0$, y la recta predice, por ejemplo, $3.0$ en ese valor extremo de $X$, el error es de $+2.0$, que al elevarse al cuadrado da un costo penalizado de $4.0$. Para reducir ese costo específico, la recta se ve obligada a rotar e inclinarse hacia arriba, moviendo la frontera de decisión hacia la derecha y provocando que puntos normales que antes se clasificaban bien como Clase 1 ahora caigan en la zona de la Clase 0 (generando Falsos Negativos).
