# Clasificación de diabetes con Streamlit

Proyecto de Minería de Datos - Tercer corte.

**Integrantes:** José Rivas, Sebastián Muñoz, Camilo Camargo y Sebastian Cervantes

## Descripción
Aplicación web que estima la clase de `Outcome` (diabetes) a partir de ocho
indicadores clínicos y demográficos, usando el pipeline entrenado en el notebook.
Dataset: Pima Indians Diabetes.

## Archivos
- `app.py`: aplicación Streamlit
- `notebook_tercer_corte_.ipynb`: entrenamiento, validación y exportación
- `pipeline_diabetes.joblib`: pipeline serializado
- `metricas_modelo.json` y `caso_prueba_conocido.json`: métricas y caso de verificación
- `diabetes.csv`: datos

## Ejecutar localmente
pip install -r requirements.txt
streamlit run app.py

## Aplicación publicada
(https://proyecto-diabetes-mineria.streamlit.app/)

## Aviso
Ejercicio académico. No sirve para diagnóstico ni decisiones clínicas.