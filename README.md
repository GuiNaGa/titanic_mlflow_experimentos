# Titanic MLflow Experimentos

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto Clase MLOPs

# Titanic — Experimentación con MLflow

## Descripción del proyecto

Este proyecto implementa un modelo de regresión logística para predecir la supervivencia de pasajeros del Titanic. Se desarrolló a partir de un notebook de exploración de datos y se convirtió en un script reutilizable de Python.

Se utilizó Cookiecutter Data Science para organizar el proyecto, un entorno virtual para aislar las dependencias y MLflow Tracking para registrar y comparar diferentes entrenamientos.

## Tecnologías utilizadas

- Python 3.14
- scikit-learn
- pandas y NumPy
- MLflow 3.17.0
- matplotlib
- Cookiecutter Data Science
- pytest

## Instalación y configuración

Clonar el repositorio:

```powershell
git clone https://github.com/GuiNaGa/titanic_mlflow_experimentos.git
cd titanic_mlflow_experimentos
```

Crear el entorno virtual:

```powershell
py -3.14 -m venv .venv
```

Activarlo en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

El entorno virtual permite utilizar las dependencias del proyecto de forma aislada. Para desactivarlo se utiliza `deactivate`.

## Ejecución del modelo

El dataset Titanic se encuentra en `data/raw/train.csv`.

Primero se debe iniciar el servidor de MLflow:

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

En otra terminal, con el entorno virtual activado, ejecutar:

```powershell
python train.py
```

También es posible modificar los hiperparámetros desde la línea de comandos:

```powershell
python train.py --C 0.1 --max_iter 1000 --solver lbfgs --seed 42
```

Los argumentos disponibles son:

- `--C`: controla la regularización.
- `--max_iter`: número máximo de iteraciones.
- `--solver`: algoritmo de optimización.
- `--seed`: semilla aleatoria para reproducibilidad.

## Seguimiento de experimentos

Las corridas se registran en MLflow bajo el experimento `Titanic_Regresion_Logistica`.

Para consultar los resultados, abrir:

http://127.0.0.1:5000

Cada corrida registra:

- Hiperparámetros utilizados.
- Métricas Accuracy, Precision, Recall y F1-score.
- Matriz de confusión como artefacto.
- Modelo entrenado mediante `mlflow.sklearn.log_model`.

## Resultados principales

Se compararon cinco configuraciones de regresión logística.

La configuración con mejor desempeño observado fue:

- C = 0.1
- max_iter = 1000
- solver = lbfgs
- seed = 42

Obtuvo un Accuracy aproximado de 0.810 y un F1-score de 0.730.

Las pruebas también mostraron que una regularización excesiva puede reducir el desempeño del modelo.

## Reproducibilidad

Se estableció la semilla aleatoria en 42 y se documentaron las dependencias en `requirements.txt`.

Dos ejecuciones con la misma configuración produjeron métricas coincidentes, lo que permitió comprobar la reproducibilidad de los resultados observados.

## Organización del proyecto

- `train.py`: entrenamiento, evaluación y registro en MLflow.
- `data/raw/train.csv`: dataset original.
- `notebooks/`: notebook de la actividad anterior.
- `requirements.txt`: dependencias con versiones.
- `tests/`: archivos para pruebas automatizadas.
- `.gitignore`: exclusión de entornos virtuales y archivos locales.

## Autor

Guillermo Navarro

Proyecto individual — Actividad de MLOps.