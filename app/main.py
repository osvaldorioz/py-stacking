from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
from stacking_module import stacking_train
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/stacking")
def calculo(samples: int):
    output_file = 'stacking.png'
    
    # Generar datos sintéticos
    np.random.seed(42)
    X = np.random.rand(samples, 2)
    y = (X[:, 0] + X[:, 1] + np.random.randn(samples) * 0.1).reshape(-1, 1)

    # Entrenar el modelo
    predictions, meta_model_predictions = stacking_train(X, y, X)

    predictions = predictions.reshape(-1, 1)  # Asegurar que sea un vector columna
    meta_model_predictions = meta_model_predictions.reshape(-1, 1)

    # Asegurar que las dimensiones coincidan
    if predictions.shape[0] != X.shape[0]:
        print(f"Dimensiones incorrectas: X.shape={X.shape}, predictions.shape={predictions.shape}")
        predictions = predictions[:X.shape[0]]

    if meta_model_predictions.shape[0] != X.shape[0]:
        print(f"Dimensiones incorrectas: X.shape={X.shape}, meta_model_predictions.shape={meta_model_predictions.shape}")
        meta_model_predictions = meta_model_predictions[:X.shape[0]]

    # Graficar dispersión de los datos
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y, label="Datos reales", alpha=0.7)    
    plt.scatter(X[:, 0], predictions, label="Predicciones del ensemble", alpha=0.7)
    plt.xlabel("Feature 1")
    plt.ylabel("Target")
    plt.legend()
    plt.title("Dispersión de datos reales vs predicciones")

    # Gráfico de comparación entre predicción del ensemble y meta-modelo
    plt.subplot(1, 2, 2)
    plt.plot(y, label="Datos reales", linestyle='dashed')
    plt.plot(predictions, label="Predicción ensemble", linestyle='solid')
    plt.plot(meta_model_predictions, label="Predicción meta-modelo", linestyle='solid')
    plt.xlabel("Índice de muestra")
    plt.ylabel("Valor Predicho")
    plt.legend()
    plt.title("Comparación de Predicciones")
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/stacking-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
