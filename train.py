
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Titanic_Regresion_Logistica"


# 2. Argumentos de línea de comandos
def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de Titanic con MLflow"
    )

    parser.add_argument(
        "--C", type=float, default=1.0,
        help="Inverso de la regularización"
    )

    parser.add_argument(
        "--max_iter", type=int, default=1000,
        help="Número máximo de iteraciones"
    )

    parser.add_argument(
        "--solver", type=str, default="lbfgs",
        choices=["lbfgs", "liblinear", "newton-cg", "saga"],
        help="Algoritmo de optimización"
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria"
    )

    return parser.parse_args()


# 3. Carga de datos
def cargar_datos():
    df = pd.read_csv(DATA_PATH)

    columnas = [
        "Pclass", "Sex", "Age", "SibSp",
        "Parch", "Fare", "Embarked"
    ]

    X = df[columnas]
    y = df["Survived"]

    return X, y


# 4. Preprocesamiento y modelo
def crear_modelo(args):

    numericas = ["Age", "SibSp", "Parch", "Fare"]
    categoricas = ["Pclass", "Sex", "Embarked"]

    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numericas
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categoricas
            )
        ]
    )

    modelo = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("clasificador", LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            solver=args.solver,
            random_state=args.seed
        ))
    ])

    return modelo


# 5. Entrenamiento, evaluación y MLflow
def entrenar():
    args = obtener_argumentos()

    if args.C <= 0:
        raise ValueError("C debe ser mayor que cero.")

    if args.max_iter <= 0:
        raise ValueError("max_iter debe ser mayor que cero.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = cargar_datos()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        # Registrar hiperparámetros
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("seed", args.seed)

        # Entrenar modelo
        modelo = crear_modelo(args)
        modelo.fit(X_train, y_train)

        # Evaluar
        predicciones = modelo.predict(X_test)

        accuracy = accuracy_score(y_test, predicciones)
        precision = precision_score(
            y_test, predicciones, zero_division=0
        )
        recall = recall_score(
            y_test, predicciones, zero_division=0
        )
        f1 = f1_score(
            y_test, predicciones, zero_division=0
        )

        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Crear matriz de confusión
        matriz = confusion_matrix(y_test, predicciones)

        display = ConfusionMatrixDisplay(
            confusion_matrix=matriz
        )

        display.plot(cmap="Blues")

        ruta_matriz = OUTPUT_DIR / "matriz_confusion.png"
        plt.savefig(ruta_matriz, bbox_inches="tight")
        plt.close()

        # Registrar artefacto
        mlflow.log_artifact(str(ruta_matriz))

        # Registrar modelo completo
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo_titanic",
            serialization_format="cloudpickle"
        )
        # Mostrar resultados
        print("\n--- Parámetros ---")
        print(f"C: {args.C}")
        print(f"max_iter: {args.max_iter}")
        print(f"solver: {args.solver}")
        print(f"seed: {args.seed}")

        print("\n--- Métricas ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        print("\nCorrida registrada correctamente en MLflow.")

        return modelo


# 6. Ejecución principal
if __name__ == "__main__":
    entrenar()
