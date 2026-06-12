import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import io
import json

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ============================================================================
st.set_page_config(
    page_title="Regresión vs Clasificación - Plataforma Educativa",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS premium para emular el diseño original (glassmorphism, colores adaptados)
st.markdown("""
<style>
    /* Estilos del contenedor principal */
    .reportview-container {
        background: #090d16;
    }
    
    /* Títulos e Info cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value-accent {
        font-size: 24px;
        font-weight: bold;
        color: #60a5fa;
    }
    .metric-value-success {
        font-size: 24px;
        font-weight: bold;
        color: #10b981;
    }
    .metric-value-warning {
        font-size: 24px;
        font-weight: bold;
        color: #e11d48;
    }
    .info-box-premium {
        background: rgba(96, 165, 250, 0.05);
        border-left: 4px solid #60a5fa;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .warning-box-premium {
        background: rgba(244, 63, 94, 0.05);
        border-left: 4px solid #f43f5e;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTES Y CONFIGURACIONES DE COLOR
# ============================================================================
COLOR_CLASS_0 = "#f43f5e" # Rosado
COLOR_CLASS_1 = "#10b981" # Verde

# ============================================================================
# DATASETS DE EJEMPLO (PRESETS MULTIVARIABLES MÓDULO 2)
# ============================================================================
CSV_DIABETES = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigree,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38.0,0.537,34,1
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1"""

CSV_ADMITIDOS = """Examen1,Examen2,NotaPreparatoria,Admitido
34.6,78.0,3.2,0
30.2,43.8,2.1,0
35.8,72.9,3.5,0
60.1,86.3,4.2,1
79.0,75.3,4.0,1
90.2,96.2,4.8,1
61.1,96.5,4.5,1
75.0,46.5,3.0,0
76.0,87.4,4.1,1
84.4,43.5,3.3,0
95.8,38.2,3.1,0
75.0,30.6,2.5,0
82.3,79.0,4.2,1
93.1,91.5,4.9,1
55.3,64.2,3.5,0"""

# PRESETS MÓDULO 1
PRESETS = {
    "Separable": pd.DataFrame([
        {"X": 0.15, "Y": 0.20, "Clase": 0},
        {"X": 0.20, "Y": 0.12, "Clase": 0},
        {"X": 0.25, "Y": 0.30, "Clase": 0},
        {"X": 0.35, "Y": 0.22, "Clase": 0},
        {"X": 0.30, "Y": 0.40, "Clase": 0},
        {"X": 0.65, "Y": 0.70, "Clase": 1},
        {"X": 0.70, "Y": 0.85, "Clase": 1},
        {"X": 0.80, "Y": 0.65, "Clase": 1},
        {"X": 0.85, "Y": 0.80, "Clase": 1},
        {"X": 0.90, "Y": 0.72, "Clase": 1}
    ]),
    "Traslapado": pd.DataFrame([
        {"X": 0.20, "Y": 0.30, "Clase": 0},
        {"X": 0.30, "Y": 0.25, "Clase": 0},
        {"X": 0.40, "Y": 0.50, "Clase": 0},
        {"X": 0.45, "Y": 0.35, "Clase": 0},
        {"X": 0.50, "Y": 0.20, "Clase": 0},
        {"X": 0.60, "Y": 0.45, "Clase": 0},
        {"X": 0.42, "Y": 0.62, "Clase": 1},
        {"X": 0.50, "Y": 0.75, "Clase": 1},
        {"X": 0.55, "Y": 0.52, "Clase": 1},
        {"X": 0.60, "Y": 0.80, "Clase": 1},
        {"X": 0.70, "Y": 0.60, "Clase": 1},
        {"X": 0.80, "Y": 0.70, "Clase": 1}
    ]),
    "Con Outliers": pd.DataFrame([
        {"X": 0.10, "Y": 0.20, "Clase": 0},
        {"X": 0.15, "Y": 0.30, "Clase": 0},
        {"X": 0.20, "Y": 0.22, "Clase": 0},
        {"X": 0.25, "Y": 0.15, "Clase": 0},
        {"X": 0.28, "Y": 0.35, "Clase": 0},
        {"X": 0.45, "Y": 0.70, "Clase": 1},
        {"X": 0.50, "Y": 0.78, "Clase": 1},
        {"X": 0.55, "Y": 0.65, "Clase": 1},
        {"X": 0.60, "Y": 0.80, "Clase": 1},
        {"X": 0.90, "Y": 0.85, "Clase": 1},
        {"X": 0.95, "Y": 0.90, "Clase": 1}
    ])
}

# ============================================================================
# ALGORITMOS MATEMÁTICOS DE REGRESIÓN LOGÍSTICA (MÓDULO 2)
# ============================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def train_logistic_regression(X_train, y_train, learning_rate, epochs):
    m, n = X_train.shape
    weights = np.zeros(n)
    bias = 0.0
    cost_history = []
    
    for epoch in range(epochs):
        z = np.dot(X_train, weights) + bias
        a = sigmoid(z)
        
        # Log-Loss con clipping
        a_clip = np.clip(a, 1e-15, 1.0 - 1e-15)
        cost = -(1 / m) * np.sum(y_train * np.log(a_clip) + (1.0 - y_train) * np.log(1.0 - a_clip))
        cost_history.append(cost)
        
        dz = a - y_train
        dw = (1 / m) * np.dot(X_train.T, dz)
        db = (1 / m) * np.sum(dz)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
    return weights, bias, cost_history

def calculate_roc_auc(y_true, probs):
    thresholds = np.linspace(1.0, 0.0, 101)
    tpr_list = []
    fpr_list = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        tn = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    # Calcular AUC con regla del trapecio
    auc = 0.0
    for i in range(1, len(tpr_list)):
        dfpr = fpr_list[i] - fpr_list[i-1]
        avgtpr = (tpr_list[i] + tpr_list[i-1]) / 2.0
        auc += dfpr * avgtpr
        
    return fpr_list, tpr_list, auc

# ============================================================================
# ESTADO GLOBAL (SESSION STATE)
# ============================================================================
# Módulo 1:
if "model_m" not in st.session_state:
    st.session_state.model_m = 0.0
if "model_b" not in st.session_state:
    st.session_state.model_b = 0.5
if "history" not in st.session_state:
    st.session_state.history = []
if "trained" not in st.session_state:
    st.session_state.trained = False

# Módulo 2:
if "multi_df" not in st.session_state:
    st.session_state.multi_df = None
if "multi_target" not in st.session_state:
    st.session_state.multi_target = ""
if "multi_features" not in st.session_state:
    st.session_state.multi_features = []
if "multi_split" not in st.session_state:
    st.session_state.multi_split = 0.8
if "multi_train_df" not in st.session_state:
    st.session_state.multi_train_df = None
if "multi_test_df" not in st.session_state:
    st.session_state.multi_test_df = None
if "multi_feature_stats" not in st.session_state:
    st.session_state.multi_feature_stats = {}
if "multi_weights" not in st.session_state:
    st.session_state.multi_weights = []
if "multi_bias" not in st.session_state:
    st.session_state.multi_bias = 0.0
if "multi_cost_history" not in st.session_state:
    st.session_state.multi_cost_history = []
if "multi_trained" not in st.session_state:
    st.session_state.multi_trained = False
if "multi_threshold" not in st.session_state:
    st.session_state.multi_threshold = 0.50

# ============================================================================
# SELECTOR DE MÓDULO PRINCIPAL
# ============================================================================
st.sidebar.title("Navegación")
app_mode = st.sidebar.radio(
    "Selecciona el Módulo a ejecutar:",
    ["Módulo 1: Simulador 2D", "Módulo 2: Regresión Logística Multivariable"],
    index=0
)

st.sidebar.markdown("---")

# ============================================================================
# MÓDULO 1: SIMULADOR 2D (REGRESIÓN LINEAL VS CLASIFICACIÓN)
# ============================================================================
if app_mode == "Módulo 1: Simulador 2D":

    def load_separable():
        st.session_state.df = PRESETS["Separable"].copy()
        st.session_state.trained = False
        st.session_state.history = []
        st.session_state.data_source_selector = "-- Seleccionar opción --"

    def load_outliers():
        st.session_state.df = PRESETS["Con Outliers"].copy()
        st.session_state.trained = False
        st.session_state.history = []
        st.session_state.data_source_selector = "-- Seleccionar opción --"

    def change_data_source():
        src = st.session_state.data_source_selector
        if src == "Traslapado (Preset)":
            st.session_state.df = PRESETS["Traslapado"].copy()
            st.session_state.trained = False
            st.session_state.history = []
        elif src == "Generar Aleatorio":
            np.random.seed(42)
            x_rand = np.random.uniform(0.1, 0.9, 15)
            labels_rand = (x_rand + np.random.normal(0, 0.1, 15) > 0.5).astype(int)
            y_rand = np.where(labels_rand == 1, np.random.uniform(0.5, 0.9, 15), np.random.uniform(0.1, 0.5, 15))
            st.session_state.df = pd.DataFrame({"X": np.round(x_rand, 3), "Y": np.round(y_rand, 3), "Clase": labels_rand})
            st.session_state.trained = False
            st.session_state.history = []

    def fit_ols(df):
        N = len(df)
        if N < 2:
            return 0.0, 0.5
        X = df["X"].values
        Y = df["Y"].values
        mean_x = np.mean(X)
        mean_y = np.mean(Y)
        num = np.sum((X - mean_x) * (Y - mean_y))
        den = np.sum((X - mean_x) ** 2)
        if den == 0:
            m = 0.0
            b = mean_y
        else:
            m = num / den
            b = mean_y - m * mean_x
        return m, b

    def calculate_metrics(df, m, b, threshold):
        N = len(df)
        if N == 0:
            return {
                "mse": 0.0, "r2": 0.0,
                "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0
            }
        X = df["X"].values
        Y = df["Y"].values
        clase = df["Clase"].values
        
        # Predicciones continuas y discretas
        y_pred_cont = m * X + b
        y_pred_disc = (y_pred_cont >= threshold).astype(int)
        
        # Regresión
        mse = np.mean((Y - y_pred_cont) ** 2)
        mean_y = np.mean(Y)
        ss_tot = np.sum((Y - mean_y) ** 2)
        ss_res = np.sum((Y - y_pred_cont) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Clasificación
        tp = np.sum((clase == 1) & (y_pred_disc == 1))
        tn = np.sum((clase == 0) & (y_pred_disc == 0))
        fp = np.sum((clase == 0) & (y_pred_disc == 1))
        fn = np.sum((clase == 1) & (y_pred_disc == 0))
        
        accuracy = (tp + tn) / N
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "mse": mse, "r2": r2,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
        }

    st.title("Plataforma Educativa: Regresión Lineal vs Clasificación")
    st.markdown(
        "Esta plataforma interactiva permite explorar el funcionamiento de la **Regresión Lineal** aplicada a problemas de **Clasificación Binaria**, "
        "analizando sus limitaciones con outliers y la importancia del umbral de decisión."
    )

    # Barra lateral del Módulo 1
    with st.sidebar:
        st.header("Configuración 2D")
        
        if "df" not in st.session_state:
            st.session_state.df = PRESETS["Separable"].copy()

        st.subheader("Ejemplos Didácticos")
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.button("Ejemplo 1: Separable", use_container_width=True, help="Caso ideal de regresión lineal", on_click=load_separable)
        with col_ex2:
            st.button("Ejemplo 2: Outliers", use_container_width=True, help="Demostración de problemas con outliers", on_click=load_outliers)

        st.subheader("Más Orígenes de Datos")
        if "data_source_selector" not in st.session_state:
            st.session_state.data_source_selector = "-- Seleccionar opción --"

        data_source = st.selectbox(
            "Otras fuentes de datos:",
            ["-- Seleccionar opción --", "Traslapado (Preset)", "Generar Aleatorio", "Cargar CSV propio"],
            key="data_source_selector",
            on_change=change_data_source
        )
        
        if st.session_state.data_source_selector == "Cargar CSV propio":
            st.markdown("**Descarga ejemplos listos para probar:**")
            try:
                with open("ejemplos_csv/datos_separables.csv", "r", encoding="utf-8") as f:
                    sep_csv = f.read()
                st.download_button("Descargar Ejemplo Separable.csv", sep_csv, "datos_separables.csv", "text/csv", use_container_width=True)
            except Exception:
                pass
                
            try:
                with open("ejemplos_csv/datos_outliers.csv", "r", encoding="utf-8") as f:
                    out_csv = f.read()
                st.download_button("Descargar Ejemplo Outliers.csv", out_csv, "datos_outliers.csv", "text/csv", use_container_width=True)
            except Exception:
                pass
                
            st.write("---")
            uploaded_file = st.file_uploader("Subir archivo CSV (debe contener columnas X, Y y Clase/Label):", type="csv")
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    cols = [c.upper() for c in uploaded_df.columns]
                    uploaded_df.columns = cols

                    col_x = [c for c in cols if "X" in c or "FEATURE" in c or "INPUT" in c]
                    col_y = [c for c in cols if "Y" in c or "OUTPUT" in c or "TARGET_CONT" in c]
                    col_class = [c for c in cols if "CLASS" in c or "CLASE" in c or "LABEL" in c or "TARGET" in c]

                    if col_x and col_class:
                        x_name = col_x[0]
                        class_name = col_class[0]
                        y_name = col_y[0] if col_y else class_name

                        df_initial = pd.DataFrame({
                            "X": uploaded_df[x_name].values,
                            "Y": uploaded_df[y_name].values,
                            "Clase": uploaded_df[class_name].values
                        })
                        df_initial["Clase"] = (df_initial["Clase"] > df_initial["Clase"].mean()).astype(int)
                        for col in ["X", "Y"]:
                            c_min, c_max = df_initial[col].min(), df_initial[col].max()
                            if c_max != c_min:
                                df_initial[col] = (df_initial[col] - c_min) / (c_max - c_min)
                            else:
                                df_initial[col] = 0.5

                        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                            st.session_state.df = df_initial
                            st.session_state.trained = False
                            st.session_state.history = []
                            st.session_state.last_uploaded_file = uploaded_file.name
                            st.success("CSV cargado con éxito.")
                            st.rerun()

                        st.subheader("Vista previa del CSV cargado")
                        st.dataframe(st.session_state.df)
                        csv_bytes = st.session_state.df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar dataset actual",
                            data=csv_bytes,
                            file_name="dataset_actual.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error("El CSV debe contener al menos una columna de entrada (X) y una columna de clase/etiqueta.")
                except Exception as e:
                    st.error(f"Error procesando el archivo CSV: {e}")

        st.sidebar.button("Reiniciar Datos a Separables", use_container_width=True, on_click=load_separable)

        st.subheader("Entrenamiento del Modelo")
        method = st.radio("Método de optimización:", ["Mínimos Cuadrados (OLS)", "Descenso de Gradiente (GD)"])
        
        gd_lr = 0.05
        gd_epochs = 100
        if method == "Descenso de Gradiente (GD)":
            gd_lr = st.slider("Tasa de aprendizaje (α):", 0.001, 0.500, 0.050, 0.001)
            gd_epochs = st.slider("Iteraciones (Épocas):", 10, 1000, 100, 10)

        st.subheader("Clasificación")
        threshold = st.slider("Umbral de decisión (T):", 0.00, 1.00, 0.50, 0.01)

        st.subheader("Opciones de Visualización")
        show_grid = st.checkbox("Mostrar Cuadrícula", value=True)
        show_residuals = st.checkbox("Mostrar Residuales (Errores)", value=True)
        show_regions = st.checkbox("Mostrar Regiones de Clasificación", value=True)

    tab_sim, tab_train, tab_eval, tab_docs = st.tabs([
        "Simulador e Interactividad",
        "Entrenamiento en Tiempo Real",
        "Evaluación de Desempeño",
        "Conceptos y Documentación"
    ])

    df = st.session_state.df

    # PESTAÑA 1: SIMULADOR
    with tab_sim:
        col_main, col_edit = st.columns([2, 1])
        
        with col_edit:
            st.subheader("Gestión de Puntos")
            st.markdown(
                "Puedes añadir nuevos registros al final de la tabla, modificar las coordenadas de X e Y "
                "(rango `0` a `1`) o alterar la clase (`0` para clase Rosada, `1` para clase Verde)."
            )
            
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "X": st.column_config.NumberColumn("Eje X (Entrada)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                    "Y": st.column_config.NumberColumn("Eje Y (Salida)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                    "Clase": st.column_config.SelectboxColumn("Clase (Etiqueta)", options=[0, 1])
                }
            )
            
            if not edited_df.equals(df):
                st.session_state.df = edited_df
                df = edited_df
                st.session_state.trained = False
                st.session_state.history = []
                st.rerun()
                
            st.info("**Consejo:** Modifica la tabla para simular distribuciones específicas, luego haz clic en 'Entrenar Modelo'.")
            
            st.write("---")
            st.subheader("Añadir Punto Rápido")
            with st.form("add_point_form", clear_on_submit=True):
                new_x = st.slider("Coordenada X:", 0.0, 1.0, 0.5, 0.01)
                new_y = st.slider("Coordenada Y:", 0.0, 1.0, 0.5, 0.01)
                new_class = st.selectbox("Clase/Etiqueta:", [0, 1])
                submitted = st.form_submit_button("Añadir Punto", use_container_width=True)
                if submitted:
                    new_row = pd.DataFrame([{"X": new_x, "Y": new_y, "Clase": new_class}])
                    st.session_state.df = pd.concat([df, new_row], ignore_index=True)
                    st.session_state.trained = False
                    st.session_state.history = []
                    st.toast(f"Punto ({new_x:.2f}, {new_y:.2f}) de Clase {new_class} añadido.")
                    st.rerun()

            if st.button("Entrenar Modelo", type="primary", use_container_width=True):
                if len(df) < 2:
                    st.error("Por favor, ingresa al menos 2 puntos para comenzar el entrenamiento.")
                else:
                    if method == "Mínimos Cuadrados (OLS)":
                        m, b = fit_ols(df)
                        st.session_state.model_m = m
                        st.session_state.model_b = b
                        st.session_state.trained = True
                        st.success("Modelo entrenado con Mínimos Cuadrados (OLS) exitosamente.")
                    else:
                        m, b = 0.0, 0.5
                        history = []
                        X = df["X"].values
                        Y = df["Y"].values
                        N = len(df)
                        for epoch in range(gd_epochs):
                            y_pred = m * X + b
                            error = y_pred - Y
                            dm = (2 / N) * np.sum(error * X)
                            db = (2 / N) * np.sum(error)
                            m -= gd_lr * dm
                            b -= gd_lr * db
                            mse_val = np.mean(error ** 2)
                            history.append({"epoch": epoch + 1, "m": m, "b": b, "mse": mse_val})
                        
                        st.session_state.model_m = m
                        st.session_state.model_b = b
                        st.session_state.history = history
                        st.session_state.trained = True
                        st.success("Modelo entrenado con Descenso de Gradiente (GD) exitosamente.")
                    st.rerun()

        with col_main:
            st.subheader("Visualización del Plano Cartesiano")
            
            m_curr = st.session_state.model_m
            b_curr = st.session_state.model_b

            fig = go.Figure()
            
            shapes = []
            if show_regions and len(df) > 0:
                if abs(m_curr) < 0.0001:
                    bg_color = COLOR_CLASS_1 if b_curr >= threshold else COLOR_CLASS_0
                    shapes.append(dict(
                        type="rect", x0=0, y0=0, x1=1, y1=1,
                        fillcolor=bg_color, opacity=0.06, layer="below", line_width=0
                    ))
                else:
                    x_front = (threshold - b_curr) / m_curr
                    if m_curr > 0:
                        x0_c0, x1_c0 = 0, max(0.0, min(1.0, x_front))
                        x0_c1, x1_c1 = max(0.0, min(1.0, x_front)), 1
                        if x1_c0 > 0:
                            shapes.append(dict(
                                type="rect", x0=x0_c0, y0=0, x1=x1_c0, y1=1,
                                fillcolor=COLOR_CLASS_0, opacity=0.06, layer="below", line_width=0
                            ))
                        if x1_c1 > x0_c1 or x0_c1 < 1:
                            shapes.append(dict(
                                type="rect", x0=x0_c1, y0=0, x1=x1_c1, y1=1,
                                fillcolor=COLOR_CLASS_1, opacity=0.06, layer="below", line_width=0
                            ))
                    else:
                        x0_c1, x1_c1 = 0, max(0.0, min(1.0, x_front))
                        x0_c0, x1_c0 = max(0.0, min(1.0, x_front)), 1
                        if x1_c1 > 0:
                            shapes.append(dict(
                                type="rect", x0=x0_c1, y0=0, x1=x1_c1, y1=1,
                                fillcolor=COLOR_CLASS_1, opacity=0.06, layer="below", line_width=0
                            ))
                        if x1_c0 > x0_c0 or x0_c0 < 1:
                            shapes.append(dict(
                                type="rect", x0=x0_c0, y0=0, x1=x1_c0, y1=1,
                                fillcolor=COLOR_CLASS_0, opacity=0.06, layer="below", line_width=0
                            ))
                            
            x_vals = np.linspace(0, 1, 100)
            y_vals = m_curr * x_vals + b_curr
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines",
                line=dict(color="#fbbf24", width=3),
                name="Regresión Lineal (y = mx + b)",
                hoverinfo="skip"
            ))
            
            if abs(m_curr) >= 0.0001:
                x_front = (threshold - b_curr) / m_curr
                if 0 <= x_front <= 1:
                    fig.add_trace(go.Scatter(
                        x=[x_front, x_front], y=[0, 1],
                        mode="lines",
                        line=dict(color="#ffffff", width=2, dash="dash"),
                        name=f"Frontera (x={x_front:.2f})",
                        hoverinfo="skip"
                    ))
                    
            if len(df) > 0:
                df["X"] = pd.to_numeric(df["X"], errors="coerce")
                df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
                df_clean = df.dropna(subset=["X", "Y"]).reset_index(drop=True)

                if show_residuals:
                    for _, row in df_clean.iterrows():
                        try:
                            pred_y = float(m_curr) * float(row["X"]) + float(b_curr)
                            fig.add_trace(go.Scatter(
                                x=[row["X"], row["X"]], y=[row["Y"], pred_y],
                                mode="lines",
                                line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dot"),
                                showlegend=False,
                                hoverinfo="skip"
                            ))
                        except Exception:
                            continue
                
                df0 = df_clean[df_clean["Clase"] == 0]
                df1 = df_clean[df_clean["Clase"] == 1]
                
                if len(df0) > 0:
                    fig.add_trace(go.Scatter(
                        x=df0["X"], y=df0["Y"],
                        mode="markers",
                        marker=dict(color=COLOR_CLASS_0, size=12, line=dict(color="white", width=1.5)),
                        name="Clase 0 (Y=0)",
                        customdata=np.stack((df0["X"], df0["Y"]), axis=-1),
                        hovertemplate="<b>Clase 0</b><br>X: %{customdata[0]:.2f}<br>Y: %{customdata[1]:.2f}<extra></extra>"
                    ))
                if len(df1) > 0:
                    fig.add_trace(go.Scatter(
                        x=df1["X"], y=df1["Y"],
                        mode="markers",
                        marker=dict(color=COLOR_CLASS_1, size=12, line=dict(color="white", width=1.5)),
                        name="Clase 1 (Y=1)",
                        customdata=np.stack((df1["X"], df1["Y"]), axis=-1),
                        hovertemplate="<b>Clase 1</b><br>X: %{customdata[0]:.2f}<br>Y: %{customdata[1]:.2f}<extra></extra>"
                    ))

            fig.update_layout(
                plot_bgcolor="#090d16",
                paper_bgcolor="#090d16",
                font=dict(color="white"),
                xaxis=dict(
                    title="Variable de Entrada (X)",
                    range=[0, 1],
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    zerolinecolor="rgba(255, 255, 255, 0.1)",
                    showgrid=show_grid
                ),
                yaxis=dict(
                    title="Salida Continua / Clase (Y)",
                    range=[0, 1],
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    zerolinecolor="rgba(255, 255, 255, 0.1)",
                    showgrid=show_grid
                ),
                margin=dict(l=40, r=40, t=40, b=40),
                shapes=shapes,
                legend=dict(
                    bgcolor="rgba(9, 13, 22, 0.8)",
                    bordercolor="rgba(255, 255, 255, 0.1)",
                    borderwidth=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.latex(rf"Ec.\ del\ modelo:\ \hat{{y}} = {m_curr:.3f}x + {b_curr:.3f}")

    # PESTAÑA 2: ANIMACIÓN GD
    with tab_train:
        st.subheader("Animación de Convergencia del Gradiente Descendiente")
        st.markdown(
            "Al utilizar **Descenso de Gradiente (GD)**, puedes observar cómo el modelo actualiza iterativamente "
            "la pendiente ($m$) y la intersección ($b$) en cada época para minimizar el error de coste (MSE)."
        )
        
        if method != "Descenso de Gradiente (GD)":
            st.info("Para ver la animación en tiempo real, selecciona el método 'Descenso de Gradiente (GD)' en la barra lateral.")
        elif len(df) < 2:
            st.warning("Agrega al menos 2 puntos en la pestaña anterior para poder entrenar el modelo.")
        else:
            col_ctrl, col_graph = st.columns([1, 2])
            
            with col_ctrl:
                st.write("**Control del Entrenamiento**")
                btn_play = st.button("Iniciar Animación de GD", use_container_width=True)
                
                status_txt = st.empty()
                params_txt = st.empty()
                loss_txt = st.empty()
                
            with col_graph:
                plot_anim_placeholder = st.empty()
                loss_curve_placeholder = st.empty()
                
            if btn_play:
                m = 0.0
                b = 0.5
                X = df["X"].values
                Y = df["Y"].values
                N = len(df)
                
                mse_history = []
                epochs_list = []
                
                for epoch in range(1, gd_epochs + 1):
                    y_pred = m * X + b
                    error = y_pred - Y
                    
                    dm = (2 / N) * np.sum(error * X)
                    db = (2 / N) * np.sum(error)
                    
                    m -= gd_lr * dm
                    b -= gd_lr * db
                    
                    mse_curr = np.mean(error ** 2)
                    mse_history.append(mse_curr)
                    epochs_list.append(epoch)
                    
                    status_txt.markdown(f"**Estado:** Entrenamiento en Curso (Época `{epoch}/{gd_epochs}`)")
                    params_txt.markdown(f"**Pesos Aprendidos:**\n* **Pendiente (m):** `{m:.4f}`\n* **Intersección (b):** `{b:.4f}`")
                    loss_txt.markdown(f"**Costo Actual (MSE):** `{mse_curr:.6f}`")
                    
                    if epoch % 2 == 0 or epoch == gd_epochs:
                        fig_anim = go.Figure()
                        
                        shapes_anim = []
                        if show_regions:
                            if abs(m) >= 0.0001:
                                x_front = (threshold - b) / m
                                if m > 0:
                                    x0_c0, x1_c0 = 0, max(0.0, min(1.0, x_front))
                                    x0_c1, x1_c1 = max(0.0, min(1.0, x_front)), 1
                                    if x1_c0 > 0:
                                        shapes_anim.append(dict(
                                            type="rect", x0=x0_c0, y0=0, x1=x1_c0, y1=1,
                                            fillcolor=COLOR_CLASS_0, opacity=0.06, layer="below", line_width=0
                                        ))
                                    if x1_c1 > x0_c1 or x0_c1 < 1:
                                        shapes_anim.append(dict(
                                            type="rect", x0=x0_c1, y0=0, x1=x1_c1, y1=1,
                                            fillcolor=COLOR_CLASS_1, opacity=0.06, layer="below", line_width=0
                                        ))
                                else:
                                    x0_c1, x1_c1 = 0, max(0.0, min(1.0, x_front))
                                    x0_c0, x1_c0 = max(0.0, min(1.0, x_front)), 1
                                    if x1_c1 > 0:
                                        shapes_anim.append(dict(
                                            type="rect", x0=x0_c1, y0=0, x1=x1_c1, y1=1,
                                            fillcolor=COLOR_CLASS_1, opacity=0.06, layer="below", line_width=0
                                        ))
                                    if x1_c0 > x0_c0 or x0_c0 < 1:
                                        shapes_anim.append(dict(
                                            type="rect", x0=x0_c0, y0=0, x1=x1_c0, y1=1,
                                            fillcolor=COLOR_CLASS_0, opacity=0.06, layer="below", line_width=0
                                        ))
                                        
                        x_line = np.linspace(0, 1, 10)
                        y_line = m * x_line + b
                        fig_anim.add_trace(go.Scatter(
                            x=x_line, y=y_line,
                            mode="lines",
                            line=dict(color="#fbbf24", width=3),
                            name=f"Época {epoch}",
                            hoverinfo="skip"
                        ))
                        
                        df0 = df[df["Clase"] == 0]
                        df1 = df[df["Clase"] == 1]
                        if len(df0) > 0:
                            fig_anim.add_trace(go.Scatter(
                                x=df0["X"], y=df0["Y"],
                                mode="markers",
                                marker=dict(color=COLOR_CLASS_0, size=10, line=dict(color="white", width=1)),
                                name="Clase 0",
                                showlegend=False,
                                hoverinfo="skip"
                            ))
                        if len(df1) > 0:
                            fig_anim.add_trace(go.Scatter(
                                x=df1["X"], y=df1["Y"],
                                mode="markers",
                                marker=dict(color=COLOR_CLASS_1, size=10, line=dict(color="white", width=1)),
                                name="Clase 1",
                                showlegend=False,
                                hoverinfo="skip"
                            ))
                        
                        fig_anim.update_layout(
                            plot_bgcolor="#090d16",
                            paper_bgcolor="#090d16",
                            font=dict(color="white", size=9),
                            xaxis=dict(
                                range=[0, 1],
                                gridcolor="rgba(255, 255, 255, 0.05)",
                                zerolinecolor="rgba(255, 255, 255, 0.1)",
                                showgrid=show_grid
                            ),
                            yaxis=dict(
                                range=[0, 1],
                                gridcolor="rgba(255, 255, 255, 0.05)",
                                zerolinecolor="rgba(255, 255, 255, 0.1)",
                                showgrid=show_grid
                            ),
                            margin=dict(l=20, r=20, t=20, b=20),
                            shapes=shapes_anim,
                            height=350
                        )
                        plot_anim_placeholder.plotly_chart(fig_anim, use_container_width=True)
                        
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            x=epochs_list, y=mse_history,
                            mode="lines",
                            line=dict(color="#60a5fa", width=2.5),
                            name="Costo (MSE)"
                        ))
                        fig_loss.update_layout(
                            plot_bgcolor="#090d16",
                            paper_bgcolor="#090d16",
                            font=dict(color="white", size=8),
                            xaxis=dict(
                                title="Épocas",
                                range=[0, gd_epochs],
                                gridcolor="rgba(255, 255, 255, 0.05)",
                                zerolinecolor="rgba(255, 255, 255, 0.1)",
                                showgrid=show_grid
                            ),
                            yaxis=dict(
                                title="Costo (MSE)",
                                gridcolor="rgba(255, 255, 255, 0.05)",
                                zerolinecolor="rgba(255, 255, 255, 0.1)",
                                showgrid=show_grid
                            ),
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=200
                        )
                        loss_curve_placeholder.plotly_chart(fig_loss, use_container_width=True)
                        
                        time.sleep(0.02)
                
                st.session_state.model_m = m
                st.session_state.model_b = b
                st.session_state.trained = True
                st.session_state.history = [{"epoch": ep, "mse": ms} for ep, ms in zip(epochs_list, mse_history)]
                status_txt.success("**¡Entrenamiento completado!** Los parámetros se han actualizado.")

    # PESTAÑA 3: EVALUACIÓN
    with tab_eval:
        st.subheader("Métricas Estadísticas del Modelo")
        st.markdown(
            "Al aplicar Regresión Lineal para Clasificación, analizamos la cercanía del ajuste continuo (Métricas de Regresión) "
            "y la calidad de la decisión final (Métricas de Clasificación)."
        )
        
        if len(df) == 0:
            st.warning("No hay datos cargados en el sistema.")
        else:
            m_curr = st.session_state.model_m
            b_curr = st.session_state.model_b
            
            metrics = calculate_metrics(df, m_curr, b_curr, threshold)
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Exactitud (Accuracy)</div>
                    <div class="metric-value-success">{metrics['accuracy']*100:.1f}%</div>
                    <small style="color:gray;">Porcentaje de aciertos</small>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Precisión (Precision)</div>
                    <div class="metric-value-accent">{metrics['precision']*100:.1f}%</div>
                    <small style="color:gray;">Exactitud de positivos predichos</small>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Sensibilidad (Recall)</div>
                    <div class="metric-value-accent">{metrics['recall']*100:.1f}%</div>
                    <small style="color:gray;">Tasa de verdaderos positivos</small>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div>F1-Score</div>
                    <div class="metric-value-accent">{metrics['f1']*100:.1f}%</div>
                    <small style="color:gray;">Medida armónica combinada</small>
                </div>
                """, unsafe_allow_html=True)
                
            col_reg, col_class = st.columns([1, 1])
            
            with col_reg:
                st.write("**Métricas del Ajuste Lineal**")
                st.info(f"**Error Cuadrático Medio (MSE):** `{metrics['mse']:.5f}`")
                st.info(f"**Coeficiente de Determinación (R²):** `{metrics['r2']:.4f}`")
                st.markdown(
                    "*(Nota: Un R² cercano a 1 indica que el modelo explica gran parte de la varianza. "
                    "Un MSE bajo representa un ajuste más cercano a las etiquetas continuas).*"
                )
                
            with col_class:
                st.write("**Matriz de Confusión**")
                
                matrix_data = pd.DataFrame(
                    [[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
                    columns=["Predicción 0", "Predicción 1"],
                    index=["Real 0", "Real 1"]
                )
                st.dataframe(matrix_data, use_container_width=True)
                
                st.markdown(
                    f"*   **Verdaderos Negativos (TN):** `{metrics['tn']}` (Clase 0 clasificados correctamente)\n"
                    f"*   **Falsos Positivos (FP):** `{metrics['fp']}` (Clase 0 clasificados como 1)\n"
                    f"*   **Falsos Negativos (FN):** `{metrics['fn']}` (Clase 1 clasificados como 0)\n"
                    f"*   **Verdaderos Positivos (TP):** `{metrics['tp']}` (Clase 1 clasificados correctamente)"
                )

    # PESTAÑA 4: DOCUMENTACIÓN Y MARCO TEÓRICO
    with tab_docs:
        st.subheader("Marco Teórico y Limitaciones de la Regresión Lineal")
        st.markdown(
            r"""
            ### Regresión Lineal para Clasificación Binaria
            La regresión lineal clásica busca encontrar una recta de la forma:
            
            $$\hat{y} = mx + b$$
            
            Que minimice el **Error Cuadrático Medio (MSE)** sobre el conjunto de entrenamiento:
            
            $$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
            
            Para usarla como clasificador binario, proyectamos el resultado de la recta y lo evaluamos contra un **umbral de decisión ($T$)**:
            
            $$\text{Clase Predicha} = \begin{cases} 1 & \text{si } \hat{y} \ge T \\ 0 & \text{si } \hat{y} < T \end{cases}$$
            
            ---
            
            ### El Impacto de los Outliers (El Problema de la Recta)
            Aunque la regresión lineal puede funcionar para clasificar cuando los datos son perfectamente separables y balanceados, sufre gravemente ante la presencia de **valores atípicos (outliers)**.
            
            **¿Por qué sucede esto?**
            1. **Función de Pérdida Cuadrática:** El MSE penaliza los errores de forma cuadrática. Si agregamos un punto muy alejado de la clase 1 en el eje X, su predicción $\hat{y}$ será muy superior a 1 (por ejemplo, 3 o 4).
            2. **Giro del Modelo:** Para intentar reducir el enorme error cuadrático de ese único punto lejano, la recta de regresión se ve forzada a rotar/inclinarse hacia arriba.
            3. **Desplazamiento de la Frontera:** Al inclinarse la recta, la frontera de decisión ($x$ donde $mx + b = T$) se desplaza lateralmente. Esto provoca que puntos cercanos que antes se clasificaban correctamente, ahora queden en el lado incorrecto de la frontera.
            
            > [!TIP]
            > **Solución:** En el aprendizaje automático real, para clasificación binaria, preferimos la **Regresión Logística**, la cual envuelve la salida lineal en una función **Sigmoide** que acota los valores estrictamente entre $[0, 1]$, evitando que los outliers lejanos deformen la frontera de decisión.
            
            ---
            
            ### Integrantes del Proyecto
            *   **Camilo Hernandez**
            *   **Fernando Vega**
            *   **Jesus Jimenez**
            
            *Proyecto adaptado a Python / Streamlit para una mejor funcionalidad educativa.*
            """
        )

# ============================================================================
# MÓDULO 2: CLASIFICACIÓN MULTIVARIABLE (REGRESIÓN LOGÍSTICA)
# ============================================================================
else:
    st.title("Regresión Logística Multivariable")
    st.markdown(
        "Este módulo permite procesar conjuntos de datos arbitrarios con múltiples variables (Features) "
        "para clasificar una etiqueta binaria (Target) mediante **Regresión Logística**. Implementa normalización, "
        "entrenamiento por gradiente, predicción en vivo y evaluación mediante curva ROC y métricas de desempeño."
    )

    # BARRA LATERAL MÓDULO 2
    with st.sidebar:
        st.header("Datos y Parámetros")
        
        multi_data_source = st.selectbox(
            "Origen del Conjunto de Datos:",
            ["-- Seleccionar opción --", "Diabetes (Salud - Preset)", "Admisiones Universitarias (Preset)", "Cargar CSV propio"]
        )

        # Cargar datos
        df_loaded = None
        if multi_data_source == "Diabetes (Salud - Preset)":
            df_loaded = pd.read_csv(io.StringIO(CSV_DIABETES))
            st.session_state.multi_df = df_loaded
            st.session_state.multi_trained = False
        elif multi_data_source == "Admisiones Universitarias (Preset)":
            df_loaded = pd.read_csv(io.StringIO(CSV_ADMITIDOS))
            st.session_state.multi_df = df_loaded
            st.session_state.multi_trained = False
        elif multi_data_source == "Cargar CSV propio":
            uploaded_file = st.file_uploader("Sube un archivo CSV:", type=["csv"])
            if uploaded_file is not None:
                try:
                    df_loaded = pd.read_csv(uploaded_file)
                    st.session_state.multi_df = df_loaded
                    st.session_state.multi_trained = False
                    st.success("CSV cargado con éxito.")
                except Exception as e:
                    st.error(f"Error al leer CSV: {e}")

        st.write("---")
        
        # Configurar Features y Target
        if st.session_state.multi_df is not None:
            df_curr = st.session_state.multi_df
            cols = list(df_curr.columns)
            
            # Autoselección por defecto de target (última columna)
            default_target_idx = len(cols) - 1
            target_var = st.selectbox("Seleccionar Variable Objetivo (Target):", cols, index=default_target_idx)
            st.session_state.multi_target = target_var
            
            # Variables de entrada (todas excepto target)
            remaining_cols = [c for c in cols if c != target_var]
            features_vars = st.multiselect(
                "Seleccionar Variables Independientes (Features):",
                remaining_cols,
                default=remaining_cols
            )
            st.session_state.multi_features = features_vars
            
            # Partición entrenamiento/prueba
            split_val = st.slider("Set de Entrenamiento (Train %):", 50, 95, 80, 5)
            st.session_state.multi_split = split_val / 100.0
            
            # Botón procesar
            if st.button("Procesar y Dividir Datos", use_container_width=True):
                if len(features_vars) == 0:
                    st.error("Debes seleccionar al menos una característica (Feature).")
                else:
                    # Shuffle y Split
                    shuffled_df = df_curr.sample(frac=1, random_state=42).reset_index(drop=True)
                    split_idx = int(len(shuffled_df) * st.session_state.multi_split)
                    
                    st.session_state.multi_train_df = shuffled_df.iloc[:split_idx].reset_index(drop=True)
                    st.session_state.multi_test_df = shuffled_df.iloc[split_idx:].reset_index(drop=True)
                    
                    # Calcular estadísticas de normalización Z-Score (en base al set de entrenamiento)
                    stats = {}
                    for col in features_vars:
                        mean_val = st.session_state.multi_train_df[col].mean()
                        std_val = st.session_state.multi_train_df[col].std()
                        if std_val == 0 or np.isnan(std_val):
                            std_val = 1.0 # Evitar división por cero
                        stats[col] = {"mean": mean_val, "std": std_val}
                    
                    st.session_state.multi_feature_stats = stats
                    st.session_state.multi_trained = False
                    st.success("Datos procesados y divididos correctamente.")
                    st.rerun()

        # Entrenamiento de Regresión Logística
        st.subheader("Entrenamiento del Modelo")
        multi_lr = st.slider("Tasa de aprendizaje (α):", 0.001, 1.000, 0.100, 0.001, format="%.3f")
        multi_epochs = st.slider("Iteraciones (Épocas):", 10, 5000, 1000, 10)

        # Botón entrenar
        if st.session_state.multi_train_df is not None:
            if st.button("Entrenar Regresión Logística", type="primary", use_container_width=True):
                train_data = st.session_state.multi_train_df
                features = st.session_state.multi_features
                target = st.session_state.multi_target
                stats = st.session_state.multi_feature_stats
                
                # Normalizar set de entrenamiento Z-Score
                X_train = np.zeros((len(train_data), len(features)))
                for idx, col in enumerate(features):
                    X_train[:, idx] = (train_data[col].values - stats[col]["mean"]) / stats[col]["std"]
                
                # Binarizar el Target basado en la media si no es binario crudo
                y_train = train_data[target].values
                if len(np.unique(y_train)) > 2 or not set(np.unique(y_train)).issubset({0, 1}):
                    mean_y = y_train.mean()
                    y_train = (y_train > mean_y).astype(int)
                    
                weights, bias, cost_history = train_logistic_regression(X_train, y_train, multi_lr, multi_epochs)
                
                st.session_state.multi_weights = list(weights)
                st.session_state.multi_bias = float(bias)
                st.session_state.multi_cost_history = cost_history
                st.session_state.multi_trained = True
                st.success("¡Modelo de Regresión Logística entrenado con éxito!")
                st.rerun()
        else:
            st.info("Primero procesa los datos para habilitar el entrenamiento.")

        st.subheader("Umbral de Clasificación")
        st.session_state.multi_threshold = st.slider("Umbral de Decisión (T):", 0.00, 1.00, 0.50, 0.01, key="threshold_multi_slider")

        # Importar Modelo
        st.subheader("Importar Modelo JSON")
        model_file = st.file_uploader("Cargar archivo de modelo (.json):", type=["json"])
        if model_file is not None:
            try:
                model_data = json.load(model_file)
                st.session_state.multi_weights = model_data["modelWeights"]
                st.session_state.multi_bias = model_data["modelBias"]
                st.session_state.multi_features = model_data["features"]
                st.session_state.multi_target = model_data["targetCol"]
                st.session_state.multi_feature_stats = model_data["featureStats"]
                st.session_state.multi_trained = True
                
                if "rawData" in model_data:
                    st.session_state.multi_df = pd.DataFrame(model_data["rawData"])
                if "trainData" in model_data:
                    st.session_state.multi_train_df = pd.DataFrame(model_data["trainData"])
                if "testData" in model_data:
                    st.session_state.multi_test_df = pd.DataFrame(model_data["testData"])
                    
                st.success("Modelo e historial cargados exitosamente desde JSON.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al cargar archivo JSON: {e}")

    # PESTAÑAS DEL MÓDULO 2
    tab_data_multi, tab_train_multi, tab_eval_multi, tab_pred_multi = st.tabs([
        "1. Carga y Configuración de Datos",
        "2. Entrenamiento del Modelo",
        "3. Evaluación de Desempeño",
        "4. Predicción Interactiva"
    ])

    # PESTAÑA 1: CARGA Y CONFIGURACIÓN DE DATOS
    with tab_data_multi:
        if st.session_state.multi_df is None:
            st.warning("Selecciona y carga un dataset en la barra lateral para comenzar.")
        else:
            df_c = st.session_state.multi_df
            st.subheader("Preview de las primeras filas del Dataset")
            st.dataframe(df_c.head(5), use_container_width=True)

            st.subheader("Estadísticas Descriptivas Generales")
            st.dataframe(df_c.describe().T, use_container_width=True)
            
            if st.session_state.multi_train_df is not None:
                st.info(
                    f"**Datos procesados e indexados:**\n"
                    f"*   **Variable Target:** `{st.session_state.multi_target}`\n"
                    f"*   **Variables Features:** `{', '.join(st.session_state.multi_features)}`\n"
                    f"*   **Registros de Entrenamiento (Train):** `{len(st.session_state.multi_train_df)}` muestras\n"
                    f"*   **Registros de Prueba (Test):** `{len(st.session_state.multi_test_df)}` muestras"
                )
                
                st.subheader("Gráfico de Dispersión Interactivo (Features)")
                st.write("Visualiza la distribución de tus datos eligiendo dos características:")
                col_sel_x, col_sel_y = st.columns(2)
                with col_sel_x:
                    feat_x = st.selectbox("Eje X:", st.session_state.multi_features, index=0)
                with col_sel_y:
                    default_idx_y = 1 if len(st.session_state.multi_features) > 1 else 0
                    feat_y = st.selectbox("Eje Y:", st.session_state.multi_features, index=default_idx_y)
                
                # Crear gráfico Plotly
                fig_disp = go.Figure()
                
                # Obtener clases reales
                target_col = st.session_state.multi_target
                classes = df_c[target_col].values
                # Binarizar por la media por coherencia visual si no es binario crudo
                if len(np.unique(classes)) > 2 or not set(np.unique(classes)).issubset({0, 1}):
                    classes = (classes > classes.mean()).astype(int)
                
                df_visual = df_c.copy()
                df_visual["Visual_Class"] = classes
                
                df0 = df_visual[df_visual["Visual_Class"] == 0]
                df1 = df_visual[df_visual["Visual_Class"] == 1]
                
                fig_disp.add_trace(go.Scatter(
                    x=df0[feat_x], y=df0[feat_y],
                    mode="markers",
                    marker=dict(color=COLOR_CLASS_0, size=9, line=dict(color="white", width=1)),
                    name="Clase 0"
                ))
                fig_disp.add_trace(go.Scatter(
                    x=df1[feat_x], y=df1[feat_y],
                    mode="markers",
                    marker=dict(color=COLOR_CLASS_1, size=9, line=dict(color="white", width=1)),
                    name="Clase 1"
                ))
                
                # Trazar frontera de decisión proyectada si el modelo está entrenado
                if st.session_state.multi_trained:
                    features_list = st.session_state.multi_features
                    idx_x = features_list.index(feat_x)
                    idx_y = features_list.index(feat_y)
                    
                    weights = st.session_state.multi_weights
                    bias = st.session_state.multi_bias
                    stats = st.session_state.multi_feature_stats
                    
                    w_x = weights[idx_x]
                    w_y = weights[idx_y]
                    
                    if w_y != 0:
                        # Ecuación de la frontera de decisión en espacio normalizado:
                        # w_x * z_x + w_y * z_y + b = 0 => z_y = -(w_x * z_x + b) / w_y
                        # Reemplazando z_x = (x - mean_x)/std_x  y  z_y = (y - mean_y)/std_y
                        min_x_val = df_c[feat_x].min()
                        max_x_val = df_c[feat_x].max()
                        x_range = np.linspace(min_x_val, max_x_val, 100)
                        
                        y_boundary = []
                        mean_x, std_x = stats[feat_x]["mean"], stats[feat_x]["std"]
                        mean_y_stat, std_y_stat = stats[feat_y]["mean"], stats[feat_y]["std"]
                        
                        for xv in x_range:
                            z_x = (xv - mean_x) / std_x
                            z_y = -(w_x * z_x + bias) / w_y
                            yv = z_y * std_y_stat + mean_y_stat
                            y_boundary.append(yv)
                            
                        fig_disp.add_trace(go.Scatter(
                            x=x_range, y=y_boundary,
                            mode="lines",
                            line=dict(color="#6366f1", width=3),
                            name="Frontera de Decisión Proyectada",
                            hoverinfo="skip"
                        ))
                
                fig_disp.update_layout(
                    plot_bgcolor="#090d16",
                    paper_bgcolor="#090d16",
                    font=dict(color="white"),
                    xaxis=dict(title=feat_x, gridcolor="rgba(255, 255, 255, 0.05)"),
                    yaxis=dict(title=feat_y, gridcolor="rgba(255, 255, 255, 0.05)"),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_disp, use_container_width=True)

    # PESTAÑA 2: ENTRENAMIENTO DEL MODELO
    with tab_train_multi:
        if not st.session_state.multi_trained:
            st.warning("El modelo aún no ha sido entrenado. Configura y entrena el modelo en la barra lateral.")
        else:
            st.subheader("Curva de Pérdida del Entrenamiento (Log-Loss)")
            st.markdown(
                "La siguiente curva ilustra el comportamiento de la función de costo de Entropía Cruzada "
                "a medida que transcurren las épocas de entrenamiento con Descenso de Gradiente."
            )
            
            cost_history = st.session_state.multi_cost_history
            fig_loss_multi = go.Figure()
            fig_loss_multi.add_trace(go.Scatter(
                x=list(range(1, len(cost_history) + 1)),
                y=cost_history,
                mode="lines",
                line=dict(color="#6366f1", width=2.5),
                name="Log-Loss (Entropy)"
            ))
            fig_loss_multi.update_layout(
                plot_bgcolor="#090d16",
                paper_bgcolor="#090d16",
                font=dict(color="white"),
                xaxis=dict(title="Épocas", gridcolor="rgba(255, 255, 255, 0.05)"),
                yaxis=dict(title="Costo (Log-Loss)", gridcolor="rgba(255, 255, 255, 0.05)"),
                margin=dict(l=40, r=40, t=40, b=40),
                height=350
            )
            st.plotly_chart(fig_loss_multi, use_container_width=True)
            st.info(f"**Costo final en la última época:** `{cost_history[-1]:.5f}`")

            st.write("---")
            st.subheader("Coeficientes y Pesos del Modelo")
            st.markdown("Pesos entrenados para cada variable predictora (Features) y sesgo del modelo:")
            
            features = st.session_state.multi_features
            weights = st.session_state.multi_weights
            bias = st.session_state.multi_bias
            
            coef_data = []
            for col, w in zip(features, weights):
                # Interpretación en lenguaje natural
                if abs(w) < 0.1:
                    interpretation = "Bajo impacto predictivo sobre la clase final."
                elif w > 0:
                    interpretation = f"El incremento de {col} incrementa la probabilidad de ser Clase 1."
                else:
                    interpretation = f"El incremento de {col} reduce la probabilidad de ser Clase 1."
                
                coef_data.append({
                    "Variable (Feature)": col,
                    "Peso (Weight)": round(w, 4),
                    "Interpretación": interpretation
                })
            
            st.table(pd.DataFrame(coef_data))
            st.info(f"**Valor del Sesgo (Bias / Intercepto):** `{bias:.4f}`")

            # Exportar Modelo JSON
            st.write("---")
            st.subheader("Exportar Modelo Entrenado")
            st.markdown("Guarda el estado actual del modelo (pesos, bias, variables y estadísticas) en un archivo JSON local.")
            
            # Construir JSON
            export_data = {
                "modelWeights": weights,
                "modelBias": bias,
                "features": features,
                "targetCol": st.session_state.multi_target,
                "featureStats": st.session_state.multi_feature_stats,
                "columns": list(st.session_state.multi_df.columns),
                "rawData": st.session_state.multi_df.to_dict(orient="records") if st.session_state.multi_df is not None else None,
                "trainData": st.session_state.multi_train_df.to_dict(orient="records") if st.session_state.multi_train_df is not None else None,
                "testData": st.session_state.multi_test_df.to_dict(orient="records") if st.session_state.multi_test_df is not None else None,
            }
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="Descargar Modelo (.json)",
                data=json_str,
                file_name=f"modelo_logistico_{st.session_state.multi_target}.json",
                mime="application/json",
                use_container_width=True
            )

    # PESTAÑA 3: EVALUACIÓN DE DESEMPEÑO
    with tab_eval_multi:
        if not st.session_state.multi_trained:
            st.warning("Entrena el modelo en la barra lateral para habilitar la evaluación.")
        elif st.session_state.multi_test_df is None:
            st.warning("No se ha procesado el set de datos de prueba.")
        else:
            test_data = st.session_state.multi_test_df
            features = st.session_state.multi_features
            target = st.session_state.multi_target
            stats = st.session_state.multi_feature_stats
            weights = np.array(st.session_state.multi_weights)
            bias = st.session_state.multi_bias
            t_val = st.session_state.multi_threshold
            
            # Normalizar set de prueba Z-Score
            X_test = np.zeros((len(test_data), len(features)))
            for idx, col in enumerate(features):
                X_test[:, idx] = (test_data[col].values - stats[col]["mean"]) / stats[col]["std"]
                
            y_test = test_data[target].values
            if len(np.unique(y_test)) > 2 or not set(np.unique(y_test)).issubset({0, 1}):
                mean_y_test = y_test.mean()
                y_test = (y_test > mean_y_test).astype(int)
                
            probs = predict_probabilities(X_test, weights, bias)
            preds = (probs >= t_val).astype(int)
            
            tp = np.sum((y_test == 1) & (preds == 1))
            tn = np.sum((y_test == 0) & (preds == 0))
            fp = np.sum((y_test == 0) & (preds == 1))
            fn = np.sum((y_test == 1) & (preds == 0))
            
            accuracy = (tp + tn) / len(y_test) if len(y_test) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Tarjetas Métricas
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Exactitud (Accuracy)</div>
                    <div class="metric-value-success">{accuracy*100:.1f}%</div>
                    <small style="color:gray;">Aciertos del set de Prueba</small>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Precisión (Precision)</div>
                    <div class="metric-value-accent">{precision*100:.1f}%</div>
                    <small style="color:gray;">Exactitud de positivos predichos</small>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Sensibilidad (Recall)</div>
                    <div class="metric-value-accent">{recall*100:.1f}%</div>
                    <small style="color:gray;">Tasa de verdaderos positivos</small>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div>F1-Score</div>
                    <div class="metric-value-accent">{f1*100:.1f}%</div>
                    <small style="color:gray;">Medida armónica combinada</small>
                </div>
                """, unsafe_allow_html=True)
                
            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.subheader("Matriz de Confusión (Set Prueba)")
                matrix_data_multi = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    columns=["Predicción 0", "Predicción 1"],
                    index=["Real 0", "Real 1"]
                )
                st.dataframe(matrix_data_multi, use_container_width=True)
                st.markdown(
                    f"*   **Verdaderos Negativos (TN):** `{tn}`\n"
                    f"*   **Falsos Positivos (FP):** `{fp}`\n"
                    f"*   **Falsos Negativos (FN):** `{fn}`\n"
                    f"*   **Verdaderos Positivos (TP):** `{tp}`"
                )
                
            with col_r:
                st.subheader("Curva ROC y Desempeño AUC")
                fpr_list, tpr_list, auc_val = calculate_roc_auc(y_test, probs)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr_list, y=tpr_list,
                    mode="lines",
                    line=dict(color="#6366f1", width=3),
                    name="Curva ROC"
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(color="rgba(255, 255, 255, 0.2)", width=1.5, dash="dash"),
                    name="Línea de Azar (AUC = 0.5)",
                    hoverinfo="skip"
                ))
                fig_roc.update_layout(
                    plot_bgcolor="#090d16",
                    paper_bgcolor="#090d16",
                    font=dict(color="white"),
                    xaxis=dict(title="Tasa de Falsos Positivos (FPR)", range=[0, 1], gridcolor="rgba(255, 255, 255, 0.05)"),
                    yaxis=dict(title="Tasa de Verdaderos Positivos (TPR)", range=[0, 1], gridcolor="rgba(255, 255, 255, 0.05)"),
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=320,
                    showlegend=False
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                st.info(f"**Área bajo la Curva ROC (AUC):** `{auc_val:.4f}`")

    # PESTAÑA 4: PREDICCIÓN INTERACTIVA
    with tab_pred_multi:
        if not st.session_state.multi_trained:
            st.warning("Entrena el modelo en la barra lateral para habilitar la predicción interactiva.")
        else:
            st.subheader("Clasificar Nueva Muestra")
            st.markdown(
                "Introduce los valores numéricos crudos para las variables predictoras elegidas "
                "y el modelo calculará la probabilidad logística en tiempo real."
            )
            
            features = st.session_state.multi_features
            stats = st.session_state.multi_feature_stats
            weights = st.session_state.multi_weights
            bias = st.session_state.multi_bias
            t_val = st.session_state.multi_threshold
            
            # Formulario dinámico de inputs
            pred_inputs = {}
            col_inputs = st.columns(min(len(features), 3))
            
            for idx, col in enumerate(features):
                col_idx = idx % 3
                with col_inputs[col_idx]:
                    # Usar la media como valor por defecto en los inputs
                    default_input_val = float(stats[col]["mean"])
                    pred_inputs[col] = st.number_input(f"Valor para {col}:", value=default_input_val, format="%.4f")
                    
            if st.button("Estimar Clase", type="primary", use_container_width=True):
                # Normalizar entradas
                X_single = np.zeros(len(features))
                for idx, col in enumerate(features):
                    X_single[idx] = (pred_inputs[col] - stats[col]["mean"]) / stats[col]["std"]
                    
                # Calcular predicción
                z_val = np.dot(X_single, weights) + bias
                prob_val = sigmoid(z_val)
                class_pred = 1 if prob_val >= t_val else 0
                
                st.write("---")
                st.subheader("Resultados de la Predicción")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    class_color = "metric-value-success" if class_pred == 1 else "metric-value-warning"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div>Clase Predicha</div>
                        <div class="{class_color}">Clase {class_pred}</div>
                        <small style="color:gray;">Bajo umbral T = {t_val:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                with col_res2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div>Probabilidad Logística (Sigmoide)</div>
                        <div class="metric-value-accent">{prob_val*100:.2f}%</div>
                        <small style="color:gray;">Confianza del estimador</small>
                    </div>
                    """, unsafe_allow_html=True)
