import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# (etiqueta, mínimo, máximo, valor por defecto, paso, ayuda)
FIELDS = {
    "Pregnancies": ("Número de embarazos", 0, 20, 1, 1,
                    "Cantidad de embarazos previos."),
    "Glucose": ("Glucosa (mg/dL)", 0, 300, 110, 1,
                "Glucosa en plasma. Use 0 si no tiene el dato."),
    "BloodPressure": ("Presión arterial diastólica (mm Hg)", 0, 200, 70, 1,
                      "Use 0 si no tiene el dato."),
    "SkinThickness": ("Grosor del pliegue del tríceps (mm)", 0, 100, 20, 1,
                      "Use 0 si no tiene el dato."),
    "Insulin": ("Insulina a las 2 horas (mu U/ml)", 0, 900, 80, 1,
                "Use 0 si no tiene el dato."),
    "BMI": ("Índice de masa corporal (IMC)", 0.0, 80.0, 25.0, 0.1,
            "Peso en kg / (estatura en m)². Use 0 si no tiene el dato."),
    "DiabetesPedigreeFunction": ("Función de antecedentes familiares", 0.0, 3.0, 0.5, 0.01,
                                 "Indicador de antecedentes familiares de diabetes."),
    "Age": ("Edad (años)", 1, 120, 30, 1, "Edad de la persona."),
}


@st.cache_resource
def load_model():
    return joblib.load(BASE / "pipeline_diabetes.joblib")


@st.cache_data
def load_json(name):
    path = BASE / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def cast(name, value):
    step = FIELDS[name][4]
    return float(value) if isinstance(step, float) else int(value)


def load_known_case():
    case = load_json("caso_prueba_conocido.json")
    if not case:
        return
    for name, value in case["features"].items():
        st.session_state[name] = cast(name, 0 if value is None else value)


st.set_page_config(page_title="Clasificación de diabetes", page_icon="🩺", layout="centered")

for name, (_, _, _, default, step, _) in FIELDS.items():
    st.session_state.setdefault(name, cast(name, default))

model = load_model()
metrics = load_json("metricas_modelo.json")

st.title("🩺 Clasificación de diabetes")
st.caption("Minería de Datos · Tercer corte · José Rivas, Sebastián Muñoz y Camilo Camargo")
st.warning("Ejercicio académico: esta herramienta **no diagnostica** ni reemplaza "
           "la valoración de un profesional de la salud.")

tab_inicio, tab_pred, tab_modelo = st.tabs(["📘 Propósito", "🔎 Predicción", "📊 Modelo"])

with tab_inicio:
    st.subheader("¿Para qué sirve?")
    st.write(
        "La aplicación estima si un perfil con ocho indicadores clínicos y demográficos "
        "se parece más a los registros con `Outcome = 1` (diabetes) o `Outcome = 0` "
        "del conjunto *Pima Indians Diabetes*, usando un modelo de aprendizaje supervisado "
        "entrenado en el proyecto."
    )
    st.subheader("¿Cómo se usa?")
    st.markdown(
        "1. Vaya a la pestaña **Predicción**.\n"
        "2. Ingrese los ocho datos. Si no conoce uno de los datos clínicos, déjelo en **0**: "
        "se tratará como dato no registrado y se completará con la mediana del entrenamiento.\n"
        "3. Pulse **Calcular** y lea la probabilidad y su interpretación."
    )
    st.info("El conjunto de datos es histórico y específico, por lo que el resultado "
            "no se puede generalizar a cualquier población.")

with tab_pred:
    st.subheader("Datos de entrada")
    st.button("Cargar caso de prueba del notebook", on_click=load_known_case)

    col1, col2 = st.columns(2)
    values = {}
    for i, name in enumerate(FEATURES):
        label, lo, hi, _, step, help_text = FIELDS[name]
        with (col1 if i % 2 == 0 else col2):
            values[name] = st.number_input(
                label, min_value=lo, max_value=hi, step=step, help=help_text, key=name
            )

    if st.button("Calcular", type="primary"):
        row = pd.DataFrame([values], columns=FEATURES).astype(float)
        row[ZERO_AS_MISSING] = row[ZERO_AS_MISSING].replace(0, np.nan)
        missing = [c for c in ZERO_AS_MISSING if pd.isna(row.at[0, c])]

        proba = float(model.predict_proba(row)[0][1])
        pred = int(model.predict(row)[0])

        st.subheader("Resultado")
        st.metric("Probabilidad estimada de clase positiva (Outcome = 1)", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

        if pred == 1:
            st.error("El perfil es **más similar al de personas con Outcome = 1** "
                     "en el conjunto de entrenamiento.")
        else:
            st.success("El perfil es **más similar al de personas con Outcome = 0** "
                       "en el conjunto de entrenamiento.")
        if missing:
            st.info("Datos no registrados (imputados con la mediana): " + ", ".join(missing))
        st.caption("Este resultado es orientativo y académico; ante dudas de salud, "
                   "consulte a un profesional.")

with tab_modelo:
    st.subheader("Modelo y métricas")
    if metrics:
        sel = metrics["selected_model"]
        res = metrics["models"][sel]
        st.write(f"**Modelo seleccionado:** {sel}")
        st.caption(metrics["selection_rule"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{res['accuracy']:.2f}")
        c2.metric("Precision", f"{res['precision_1']:.2f}")
        c3.metric("Recall", f"{res['recall_1']:.2f}")
        c4.metric("F1", f"{res['f1_1']:.2f}")
        st.write(f"F1 en validación cruzada: {res['cv_f1_mean']:.3f} "
                 f"(± {res['cv_f1_std']:.3f})")
        tn, fp = res["confusion_matrix"][0]
        fn, tp = res["confusion_matrix"][1]
        st.dataframe(
            pd.DataFrame([[tn, fp], [fn, tp]],
                         index=["Real 0", "Real 1"],
                         columns=["Predicho 0", "Predicho 1"])
        )
        st.caption(f"Registros: {metrics['dataset_rows']} · "
                   f"Prueba: {int(metrics['test_size']*100)}% · "
                   f"Validación cruzada: {metrics['cross_validation_folds']} particiones")
    else:
        st.write("No se encontró `metricas_modelo.json`.")