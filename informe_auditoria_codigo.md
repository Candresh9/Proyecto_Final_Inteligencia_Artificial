# 🔍 Informe de Auditoría de Código y Arquitectura del Sistema
**Proyecto:** Regresión Lineal vs Clasificación / Regresión Logística Multivariable
**Autores del Repositorio:** Camilo Hernandez, Fernando Vega, Jesus Jimenez
**Rol del Auditor:** Auditor de Software & Ingeniero de Machine Learning Senior
**Propósito:** Desglose técnico exhaustivo de los componentes, funciones y fundamentos matemáticos del proyecto para facilitar su estudio académico y preparación de sustentación.

---

## 1. Arquitectura General del Repositorio

El repositorio presenta una arquitectura híbrida de **doble plataforma**. Originalmente diseñado como una aplicación web interactiva basada en tecnologías de frontend tradicionales (HTML/CSS/JS), posteriormente se portó a una aplicación interactiva nativa de **Python** utilizando **Streamlit**. Esto permite comparar dos aproximaciones de implementación de software educativo para Machine Learning.

### Estructura de Archivos Auditorada

| Archivo | Tecnología | Rol en la Arquitectura | Descripción General |
| :--- | :--- | :--- | :--- |
| **`app.py`** | Python 3.9+ / Streamlit | Aplicación de Backend/Frontend Unificado | Orquesta la versión moderna del simulador en Python. Utiliza Pandas para la manipulación de datos, NumPy para el cómputo lineal, Matplotlib/Plotly para gráficos interactivos y Streamlit para la interfaz reactiva. |
| **`app.js`** | JavaScript (ES6) | Lógica de Frontend Original | Contiene toda la lógica de simulación matemática (OLS, Descenso de Gradiente 2D) y el motor completo de Regresión Logística Multivariable (con normalización, entrenamiento iterativo, predicción y evaluación ROC/AUC). |
| **`index.html`** | HTML5 / CSS3 | Interfaz de Usuario Original | Define la estructura visual y los contenedores del simulador 2D y del panel de control multivariable para la versión Web. |
| **`style.css`** | CSS3 (Custom) | Diseño Visual Web | Estilos CSS personalizados para la interfaz del navegador, utilizando esquemas oscuros, botones interactivos y layouts flexibles. |
| **`requirements.txt`**| Configuración | Gestión de Dependencias | Declara las librerías necesarias para ejecutar `app.py` (`streamlit`, `pandas`, `numpy`, `matplotlib`, `plotly`). |
| **`DOCUMENTACION.md`**| Markdown | Documentación Académica | Explica la justificación de la migración a Python y detalla el problema didáctico de los outliers en la Regresión Lineal. |

---

## 2. Auditoría Detallada del Backend en Python: `app.py`

Este archivo contiene la versión nativa en Python. Utiliza una arquitectura declarativa donde la UI se renderiza reactivamente ante cualquier cambio de estado almacenado en `st.session_state`.

### 📊 Variables de Estado Globales (`st.session_state`)
* **`st.session_state.df`**: Dataframe de Pandas (`pd.DataFrame`) que almacena las coordenadas de los puntos $(X, Y)$ y sus etiquetas de clase ($0$ o $1$).
* **`st.session_state.model_m` / `st.session_state.model_b`**: Parámetros actuales del modelo de la recta de regresión ($y = mx + b$).
* **`st.session_state.history`**: Historial del coste $MSE$ y coeficientes por época durante el entrenamiento con Descenso de Gradiente.
* **`st.session_state.trained`**: Bandera booleana que indica si el modelo ha completado su entrenamiento.

### ⚙️ Desglose de Funciones

#### 1. `change_data_source()`
* **Propósito:** Manejar el cambio de origen de datos seleccionado por el usuario en el sidebar.
* **Operación:**
  * Si se selecciona `"Traslapado (Preset)"`, copia la matriz predefinida de puntos traslapados al estado del dataframe local y desactiva el estado entrenado.
  * Si se selecciona `"Generar Aleatorio"`, inicializa una semilla aleatoria (`np.random.seed(42)`), genera $15$ muestras uniformes en $X$, calcula etiquetas binarias aplicando ruido normal en un umbral de $0.5$, asigna coordenadas $Y$ congruentes en rangos diferenciados para cada clase y carga el conjunto en la sesión.
* **Código Clave:**
  ```python
  labels_rand = (x_rand + np.random.normal(0, 0.1, 15) > 0.5).astype(int)
  y_rand = np.where(labels_rand == 1, np.random.uniform(0.5, 0.9, 15), np.random.uniform(0.1, 0.5, 15))
  ```

#### 2. `fit_ols(df)`
* **Propósito:** Calcular la pendiente ($m$) y el intercepto ($b$) de la recta de regresión lineal por el método analítico de Mínimos Cuadrados Ordinarios (OLS - Ordinary Least Squares).
* **Fórmula Matemática Implícita:**
  $$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} \qquad b = \bar{y} - m\bar{x}$$
* **Operación:** Extrae los vectores $X$ e $Y$ del DataFrame, calcula sus medias, implementa la covarianza y varianza muestral de forma vectorizada usando NumPy y maneja la división por cero si todos los puntos tienen el mismo valor de $X$.

#### 3. `calculate_metrics(df, m, b, threshold)`
* **Propósito:** Generar todas las métricas de regresión (MSE, $R^2$) y clasificación (Matriz de confusión, Accuracy, Precision, Recall, F1-Score) basándose en las predicciones continuas umbralizadas.
* **Operación:**
  * Calcula las predicciones continuas: $\hat{y}_i = m \cdot x_i + b$.
  * Umbraliza las predicciones continuas para obtener predicciones binarias discretas: $\text{Clase Predicha} = 1$ si $\hat{y}_i \ge \text{threshold}$, sino $0$.
  * Calcula el **Error Cuadrático Medio (MSE)** como la media de los residuales al cuadrado: $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$.
  * Calcula el **Coeficiente de Determinación ($R^2$)** usando la suma de cuadrados de los residuos y la suma total de cuadrados.
  * Evalúa la matriz de confusión comparando las predicciones discretas frente a las clases reales para contar Verdaderos Positivos (TP), Verdaderos Negativos (TN), Falsos Positivos (FP) y Falsos Negativos (FN).
  * Devuelve un diccionario estructurado de métricas.

---

## 3. Auditoría Detallada del Frontend en JavaScript: `app.js`

El archivo `app.js` está estructurado en dos grandes módulos lógicos:
1. **Módulo 1:** Simulador Interactivo 2D en Canvas HTML5 para la comparación interactiva de Regresión Lineal vs Clasificación.
2. **Módulo 2:** Panel de Control de Regresión Logística Multivariable que procesa archivos CSV subidos por el usuario, entrena redes con Descenso de Gradiente multivariable en el cliente y evalúa curvas ROC.

---

### 🧱 MÓDULO 1: Simulador Interactivo 2D

Maneja el renderizado de gráficos directamente en la GPU del navegador mediante la API de Canvas 2D.

#### 1. Coordenadas y Escalas Matemáticas

* **`getCanvasScales()`**
  * **Propósito:** Determinar los límites útiles en píxeles del lienzo (Canvas) para evitar que los puntos se dibujen sobre los ejes del plano o se recorten en las esquinas. Devuelve límites horizontales (`minX`, `maxX`) y verticales (`minY`, `maxY`) considerando un margen interno (`padding = 50px`).
* **`toModelCoords(canvasX, canvasY)`**
  * **Propósito:** Conversión de coordenadas de la pantalla (píxeles del cursor del mouse) a coordenadas del modelo matemático real (rango $[0, 1]$).
  * **Ecuación matemática:**
    $$x_{\text{model}} = \frac{x_{\text{canvas}} - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}} \qquad y_{\text{model}} = \frac{y_{\text{min}} - y_{\text{canvas}}}{y_{\text{min}} - y_{\text{max}}}$$
* **`toCanvasCoords(modelX, modelY)`**
  * **Propósito:** Operación inversa a la anterior. Convierte una coordenada matemática en el rango $[0, 1]$ a píxeles exactos en el canvas para el renderizado físico de puntos y rectas.

#### 2. Control de Entrada e Interactividad del Plano

* **`findPointIndexAt(canvasX, canvasY)`**
  * **Propósito:** Detección de colisión por proximidad física (distancia euclidiana) entre el cursor y cualquier punto registrado en el lienzo. Usa la fórmula:
    $$\text{Distancia} = \sqrt{(x_{\text{point}} - x_{\text{cursor}})^2 + (y_{\text{point}} - y_{\text{cursor}})^2}$$
    Si la distancia es menor o igual al radio del punto más un margen de tolerancia ($14\text{px}$), devuelve el índice del punto en el arreglo.
* **`handleCanvasMouseDown(e)`**
  * **Propósito:** Gestor del evento de clic inicial sobre el canvas.
  * **Operación:** Si el usuario hace clic izquierdo sobre un punto existente, lo marca para arrastre (`draggedIndex`). Si hace clic izquierdo en una zona vacía del plano, crea dinámicamente un nuevo punto con la clase activa y ejecuta la reestimación automática de la recta.
* **`handleCanvasMouseMove(e)`**
  * **Propósito:** Gestor del movimiento del cursor sobre el canvas.
  * **Operación:** Si `draggedIndex` está activo, traduce la posición del cursor a coordenadas $[0, 1]$ y actualiza el punto en tiempo real, forzando un redibujado completo del lienzo. Si no arrastra nada, altera el puntero del mouse a `pointer` (mano) o `crosshair` (retícula) según si está flotando sobre un punto.
* **`handleCanvasContextMenu(e)`**
  * **Propósito:** Interceptar el clic derecho sobre el canvas para alternar la etiqueta de la clase ($0 \leftrightarrow 1$) del punto sobre el que se hace clic.
* **`handleCanvasDblClick(e)`**
  * **Propósito:** Eliminar un punto del dataset cuando el usuario hace doble clic sobre él.

#### 3. Algoritmos de Ajuste de Recta 2D

* **`fitOLS()`**
  * **Propósito:** Implementa el estimador analítico de Mínimos Cuadrados Ordinarios en JavaScript. Calcula los coeficientes de la recta $m$ y $b$ de manera instantánea y actualiza las métricas en la interfaz.
* **`startSimulatorGDTraining()`**
  * **Propósito:** Inicia un bucle asíncrono controlado por `setInterval` (cada $25\text{ms}$) para simular visualmente la optimización interactiva de la recta por Descenso de Gradiente (GD).
  * **Operación:** Inicializa $m = 0.0$ y $b = 0.5$ (recta horizontal intermedia) y en cada iteración invoca `performGradientDescentStep2D()`, redibuja la recta y actualiza el gráfico de decaimiento del coste ($MSE$) para que el estudiante perciba la velocidad de convergencia.
* **`performGradientDescentStep2D()`**
  * **Propósito:** Ejecutar un paso de optimización (actualización de pesos) usando derivadas parciales de la función de costo del Error Cuadrático Medio ($MSE$).
  * **Derivación Matemática:**
    El costo es $J(m, b) = \frac{1}{N}\sum_{i=1}^N (mx_i + b - y_i)^2$. Sus derivadas respecto a los parámetros son:
    $$\frac{\partial J}{\partial m} = \frac{2}{N} \sum_{i=1}^N (mx_i + b - y_i) \cdot x_i$$
    $$\frac{\partial J}{\partial b} = \frac{2}{N} \sum_{i=1}^N (mx_i + b - y_i)$$
    Las actualizaciones de los parámetros son:
    $$m \leftarrow m - \alpha \frac{\partial J}{\partial m} \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
  * **Implementación:** El código implementa esta sumatoria acumulando los gradientes tras recorrer el arreglo de puntos y los multiplica por el factor de tasa de aprendizaje (`gdLearningRate`).

#### 4. Motor de Renderizado Visual (Canvas 2D)

* **`render()`**
  * **Propósito:** Orquestar el dibujado secuencial del lienzo. Limpia el canvas, sombrea las regiones de decisión, dibuja las guías de la cuadrícula, traza los ejes cartesianos con sus valores, proyecta las líneas de error (residuales), traza la recta ajustada, marca la frontera de decisión y posiciona las esferas de datos de las clases.
* **`drawClassificationRegions(scales)`**
  * **Propósito:** Sombrear de color rosado translúcido la zona asignada a la Clase 0, y de verde la zona asignada a la Clase 1.
  * **Operación:** Encuentra el punto de cruce en X donde el modelo cruza el umbral ($T$):
    $$x_{\text{front}} = \frac{T - b}{m}$$
    Dibuja rectángulos de fondo en el canvas desde $x = 0$ hasta $x_{\text{front}}$ y desde $x_{\text{front}}$ hasta $x = 1$, pintándolos con el color de su respectiva clase.
* **`drawResidualLines(scales)`**
  * **Propósito:** Dibujar líneas verticales punteadas que conectan la coordenada Y real de cada punto con la coordenada Y proyectada por la recta ($\hat{y} = mx + b$). Representa visualmente la magnitud del error que el optimizador intenta minimizar.

---

### 🧬 MÓDULO 2: Clasificación Multivariable (Regresión Logística)

Módulo enfocado al procesamiento de datos tabulares arbitrarios ($N$-dimensionales) para clasificación binaria aplicando la Regresión Logística clásica.

#### 1. Procesamiento e Ingesta de Datos

* **`parseCSV(csvText)`**
  * **Propósito:** Convertidor e intérprete de archivos CSV planos subidos por el usuario en texto crudo.
  * **Operación:**
    * Separa la primera línea para obtener las cabeceras (nombres de variables/columnas).
    * Parsea cada fila subsiguiente validando que contenga números flotantes válidos.
    * Puebla dinámicamente el selector de variable objetivo (`targetCol`), inicializa estadísticas descriptivas del conjunto y autoconfigura los checkboxes de variables independientes (características o *features*).
* **`calculateDescriptiveStats()`**
  * **Propósito:** Calcular métricas estadísticas clave para cada columna del CSV cargado: media ($\mu$), desviación estándar ($\sigma$), mínimo y máximo. Estas métricas son fundamentales para la normalización del modelo.
* **`processAndSplitData()`**
  * **Propósito:** Particionar el dataset en dos subconjuntos independientes: Entrenamiento (Train) y Prueba (Test).
  * **Operación:**
    * Filtra qué variables fueron seleccionadas en la interfaz como *features*.
    * Desordena de manera aleatoria (*shuffle*) los datos para eliminar cualquier sesgo de ordenación inicial:
      ```javascript
      const shuffled = [...rawData].sort(() => 0.5 - Math.random());
      ```
    * Realiza el particionamiento basado en la proporción seleccionada por el usuario (por defecto $80\%$ entrenamiento y $20\%$ prueba).
    * Dispara la normalización y reconstruye la interfaz de entrada para predicciones en vivo.

#### 2. Normalización Z-Score de Datos
La Regresión Logística requiere que los datos estén normalizados para evitar que variables con escalas grandes (ej. Salarios de 5 cifras) dominen a variables pequeñas (ej. Edad de 2 cifras) y para asegurar la convergencia del Descenso de Gradiente.

* **`calculateNormalizationStats()`**
  * **Propósito:** Calcula la media ($\mu$) y desviación estándar ($\sigma$) de cada característica seleccionada **únicamente sobre el conjunto de entrenamiento** (para evitar la filtración de datos o *data leakage*).
* **`normalize(value, featureName)`**
  * **Propósito:** Aplica la estandarización clásica Z-Score al valor de entrada:
    $$z = \frac{x - \mu}{\sigma}$$

#### 3. Algoritmo de Entrenamiento de Regresión Logística

* **`sigmoid(z)`**
  * **Propósito:** Implementar la función de activación logística que mapea cualquier número real al intervalo abierto $(0, 1)$, interpretado como una probabilidad.
  * **Fórmula:**
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **`trainLogisticRegression()`**
  * **Propósito:** Ajustar los pesos vectoriales $\mathbf{w}$ y el sesgo $b$ minimizando la función de pérdida de **Entropía Cruzada Binaria (Log-Loss)** mediante Descenso de Gradiente de Lote Completo (Batch Gradient Descent).
  * **Fundamento Matemático del Optimizador:**
    Para un conjunto de $M$ muestras y $N$ características:
    1. **Predicción Logística:** Para cada muestra $i$, calcula la combinación lineal y le aplica la sigmoide:
       $$z^{(i)} = b + \sum_{j=1}^N w_j \cdot x_j^{(i)} \qquad a^{(i)} = \sigma(z^{(i)})$$
    2. **Función de Costo (Log-Loss):**
       $$J(\mathbf{w}, b) = -\frac{1}{M} \sum_{i=1}^M \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$
       *Nota: Para evitar errores numéricos por indeterminación en $\log(0)$, las predicciones $a^{(i)}$ se acotan (clip) al intervalo $[10^{-15}, 1 - 10^{-15}]$.*
    3. **Cálculo de Gradientes (Derivadas Parciales):**
       $$\frac{\partial J}{\partial w_j} = \frac{1}{M} \sum_{i=1}^M (a^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$
       $$\frac{\partial J}{\partial b} = \frac{1}{M} \sum_{i=1}^M (a^{(i)} - y^{(i)})$$
    4. **Actualización de Parámetros:**
       $$w_j \leftarrow w_j - \alpha \frac{\partial J}{\partial w_j} \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
  * **Operación:** Itera el número de épocas configuradas por el estudiante actualizando los pesos y registrando el costo histórico en cada ciclo para graficarlo al final.

#### 4. Evaluación de Rendimiento Avanzada

* **`evaluateModel(thresholdVal)`**
  * **Propósito:** Evaluar el modelo de clasificación multivariable sobre el conjunto de prueba (Test) no visto durante el entrenamiento. Genera la matriz de confusión y las métricas Accuracy, Precision, Recall y F1-Score utilizando el umbral seleccionado por el usuario.
* **`calculateAndRenderROC()`**
  * **Propósito:** Calcular e ilustrar la **Curva ROC (Receiver Operating Characteristic)** y estimar el **AUC (Area Under the Curve)**.
  * **Operación matemática:**
    * Calcula la probabilidad predicha para cada muestra en el set de prueba.
    * Itera el umbral de decisión $T$ desde $0.0$ hasta $1.0$ con pasos de $0.01$.
    * Para cada umbral, calcula la Tasa de Verdaderos Positivos (TPR - Sensibilidad) y la Tasa de Falsos Positivos (FPR - $1 - \text{Especificidad}$):
      $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} \qquad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
    * Registra estos pares de puntos $(\text{FPR}, \text{TPR})$ y traza la curva en la interfaz usando Chart.js.
    * Estima el **AUC** integrando numéricamente el área bajo la curva mediante la regla del trapecio sobre los puntos ordenados:
      $$\text{AUC} \approx \sum_{k=1}^K \frac{(\text{TPR}_k + \text{TPR}_{k-1})}{2} \cdot (\text{FPR}_k - \text{FPR}_{k-1})$$

#### 5. Predicción y Persistencia

* **`makePrediction()`**
  * **Propósito:** Permitir que el usuario ingrese valores arbitrarios en tiempo real para las características del modelo en un formulario web dinámico, los normalice utilizando las estadísticas de entrenamiento guardadas y devuelva la probabilidad estimada y la clase asignada.
* **`exportModel()`**
  * **Propósito:** Guardar el estado completo del modelo en un archivo de texto JSON descargable para que pueda ser restaurado posteriormente sin tener que reentrenar.
  * **Campos exportados:** Pesos de características (`modelWeights`), intercepto/sesgo (`modelBias`), lista de características seleccionadas, nombre de variable objetivo, parámetros de normalización ($\mu$ y $\sigma$), conjunto de datos de entrenamiento y prueba y estadísticas descriptivas del conjunto.

---

## 4. Análisis Crítico de Machine Learning e Implicaciones de Código

Como auditores, identificamos tres áreas clave en el código que demuestran los fundamentos teóricos del Machine Learning y exponen riesgos algorítmicos típicos:

### ⚠️ 1. El Problema de los Outliers en Regresión Lineal para Clasificación
El código del simulador 2D ilustra perfectamente por qué la **Regresión Lineal no es adecuada para clasificación**.
* **Mecánica del error:** La función de costo de la Regresión Lineal (el MSE en `calculateMSE()`) penaliza la distancia vertical al cuadrado. Si se introduce un outlier de Clase 1 muy lejano a la derecha del plano ($X \to 1.0$), la recta intentará dar una predicción $\hat{y}$ mucho mayor que $1.0$ (por ejemplo, $2.5$).
* **Consecuencia:** Como la distancia $(2.5 - 1.0)^2 = 2.25$ es fuertemente castigada, el optimizador rotará la recta hacia abajo para acercarla al outlier, reduciendo la pendiente. Al rotar la recta, la frontera de decisión ($x_{\text{front}} = \frac{T - b}{m}$) se desliza drásticamente hacia la derecha. Esto causa que puntos intermedios que pertenecen a la Clase 1 ahora sean clasificados incorrectamente como Clase 0, generando múltiples **Falsos Negativos** (puntos verdes en la zona sombreada rosada).

### 🛡️ 2. Resistencia de la Regresión Logística a los Outliers
El módulo 2 solventa este problema encapsulando la combinación lineal en la función **Sigmoide** (`sigmoid`).
* **Mecánica:** Sin importar lo alejado que esté un punto en el eje de características, la salida de la sigmoide estará estrictamente acotada en el rango $(0, 1)$. Una muestra con un valor extremadamente grande tendrá una probabilidad $\approx 1.0$, y su costo logarítmico (en `trainLogisticRegression()`) será prácticamente cero, eliminando la necesidad del modelo de rotar o ajustar su frontera de decisión para acomodar este punto extremo. Esto mantiene estable la frontera de decisión para el resto de muestras cercanas.

### 🔍 3. Riegos y Fallos Potenciales Detectados en la Auditoría de Código
* **División por cero en Normalización:** En `normalize(value, featureName)`, si una característica en el dataset subido por el usuario es constante (todas las filas tienen el mismo valor), su desviación estándar $\sigma$ calculada en `calculateNormalizationStats` será $0.0$. El código cuenta con una salvaguarda de respaldo asignando `std = 1` si el cálculo da cero:
  ```javascript
  const std = Math.sqrt(sumSq / trainData.length) || 1;
  ```
  Esto evita que el sistema retorne `NaN` durante el entrenamiento. Sin embargo, una característica con varianza cero no aporta poder predictivo y debería ser descartada.
* **Sesgo por Datos no Balanceados:** Al calcular el AUC y la curva ROC en `calculateAndRenderROC()`, si el dataset de prueba no contiene muestras de alguna de las clases, las tasas TPR o FPR tendrán indeterminaciones matemáticas por división por cero. Aunque el código maneja esto con un operador ternario (`(tp_t + fn_t) > 0 ? ... : 0`), el reporte resultante del AUC podría no ser estadísticamente representativo de la realidad del modelo.

---

## 5. Resumen de la Auditoría y Recomendaciones Académicas

El código auditado está escrito de manera muy didáctica, separando claramente la manipulación del DOM, la renderización visual en Canvas y los cálculos numéricos en crudo.

### 💡 Puntos Clave para Estudiar:
1. **Compara las dos formas de optimizar:**
   * En OLS (`fitOLS()` / `fit_ols(df)`), el ajuste se realiza en un único paso de tiempo ($O(M)$) usando álgebra directa.
   * En Descenso de Gradiente (`performGradientDescentStep2D` y `trainLogisticRegression`), el ajuste es iterativo ($O(\text{épocas} \times M \times N)$) y depende directamente de los hiperparámetros de tasa de aprendizaje ($\alpha$) y número de iteraciones.
2. **Entiende la diferencia entre Métricas Continuas y Discretas:**
   * El MSE y el $R^2$ evalúan qué tan bien la recta sigue la tendencia del plano continuo.
   * La matriz de confusión, precisión, sensibilidad y F1 evalúan la calidad de las decisiones categóricas tras aplicar el umbral $T$.

Este análisis detallado te proporcionará una sólida comprensión del funcionamiento interno del proyecto para cualquier evaluación académica o presentación técnica.
