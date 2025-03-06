El **Stacking Approach** es una técnica de ensamblado de modelos que combina múltiples predictores (base learners) y utiliza un meta-modelo para hacer la predicción final. Se entrena en dos fases:  

1. **Fase 1 - Modelos Base:**  
   - Se entrenan múltiples modelos (por ejemplo, regresores o clasificadores).  
   - Cada modelo genera predicciones sobre los datos de entrenamiento.  
   - Estas predicciones se usan como nuevas características para entrenar un meta-modelo.  

2. **Fase 2 - Meta-Modelo:**  
   - Recibe como entrada las predicciones de los modelos base.  
   - Aprende a combinar esas predicciones para mejorar el rendimiento global.  

### **Implementación en este programa**  
- Se genera un conjunto de datos sintético.  
- Se entrena un modelo de stacking mediante `train_stacking()`, que implementa la combinación de modelos base y el meta-modelo.  
- Se comparan las predicciones de los modelos base con el meta-modelo.  
- Se generan dos gráficos:  
  1. **Dispersión de datos reales vs predicciones** para visualizar la calidad del ajuste.  
  2. **Comparación de predicciones** entre los modelos base y el meta-modelo para evaluar su desempeño.  

Este enfoque permite mejorar la precisión del modelo al aprovechar las fortalezas de múltiples predictores. 
