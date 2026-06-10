# Plataforma Completa de Clasificación (Regresión Logística)

Este repositorio contiene el Proyecto Final de la asignatura de Inteligencia Artificial desarrollado por **Camilo Hernandez, Fernando Vega y Jesus Jimenez**.

La aplicación es una plataforma web interactiva de **Machine Learning** que ejecuta todo su procesamiento (parseo de datos, separación, entrenamiento matemático, predicción) directamente en el navegador del usuario utilizando JavaScript puro, sin requerir un backend.

## Cumplimiento de Objetivos Específicos (Rúbrica)

La aplicación demuestra la comprensión integral del ciclo de vida de un modelo supervisado:

*   **A. Estructuración del Problema:** La pestaña *Datos* permite cargar un dataset real (CSV) y seleccionar dinámicamente qué columna representa el objetivo a clasificar (Target) y cuáles son las características predictoras (Features).
*   **B. Separación de Datos (Train/Test Split):** Antes del entrenamiento, el sistema baraja (shuffles) y separa el dataset según un porcentaje definido por el usuario (ej. 80% entrenamiento, 20% prueba) para evitar el sobreajuste (overfitting).
*   **C. Entrenamiento de Regresión Logística:** El modelo se entrena en la pestaña *Entrenamiento* utilizando el algoritmo de **Gradiente Descendiente Multivariable**. Se optimiza la función de pérdida *Log-Loss* (Entropía Cruzada) aplicando la función de activación Sigmoide a la combinación lineal de los datos previamente normalizados mediante *Z-Score*.
*   **D. Interpretación de Coeficientes:** Una vez finalizado el entrenamiento, el sistema despliega una tabla con los pesos ($W$) aprendidos para cada característica e interpreta automáticamente si influyen positiva o negativamente hacia la clase 1.
*   **E. Evaluación del Modelo:** En la pestaña *Evaluación*, se calculan las proyecciones del modelo sobre los datos nunca antes vistos (Conjunto de Prueba). Se genera la **Matriz de Confusión** y se derivan métricas como *Accuracy*, *Precision*, *Recall* y *F1-Score*.
*   **F. Predicción con Nuevos Datos:** La pestaña *Predicciones* lee dinámicamente el esquema de variables del CSV cargado y genera un formulario. Permite introducir nuevos registros manuales, normalizarlos bajo la escala del modelo, y retornar una probabilidad y clasificación final.
*   **G. Despliegue en la Nube:** Al ser una arquitectura 100% frontend (*Single Page Application* en HTML/CSS/JS), está alojada en **GitHub Pages**. Esto significa que está desplegada de forma estable y accesible públicamente sin requerir mantenimiento de servidores.
*   **H. Uso de IA Generativa:** Documentado internamente en la pestaña "Documentación IA" de la propia plataforma, donde se especifica que se utilizó Google DeepMind Antigravity como copiloto para estructurar la UI, corregir las matemáticas del gradiente multivariable y afianzar los conceptos matemáticos del estudiante.

---

## Despliegue en la Nube

🔗 **[Visita la aplicación web funcional aquí](https://Candresh9.github.io/Proyecto_Final_Inteligencia_Artificial/)**
*(Nota: Asegúrate de habilitar GitHub Pages en la rama `main` en la configuración del repositorio).*

## Ejecución Local

Si prefieres ejecutarlo localmente:
1. Clona el repositorio:
   `git clone https://github.com/Candresh9/Proyecto_Final_Inteligencia_Artificial.git`
2. Abre el archivo `index.html` en tu navegador favorito.
