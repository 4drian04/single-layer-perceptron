# 🧠 Perceptrón Simple para Clasificación de Aprobado/Suspenso

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-✔️-orange)
![Pandas](https://img.shields.io/badge/Pandas-✔️-purple)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-✔️-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

Este proyecto implementa desde cero un **Perceptrón** para clasificación binaria y lo compara con modelos de *Machine Learning* como la **Regresión Logística**.

El objetivo es predecir si un estudiante **aprueba o suspende** en función de distintas variables académicas.

---

## 📊 Dataset

El conjunto de datos contiene información sobre estudiantes:

- `attendance_pct`: porcentaje de asistencia  
- `homework_pct`: nota de tareas  
- `midterm_score`: nota de mitad de curso  
- `study_hours_per_week`: horas de estudio semanales  
- `pass`: variable objetivo (1 = aprueba, 0 = suspende)

---

## ⚙️ Implementación del Perceptrón

El perceptrón ha sido implementado desde cero con:

- Inicialización de pesos en 0  
- Función de activación **sigmoide**  
- Actualización iterativa de pesos  
- Parámetros configurables:
  - `eta` (learning rate)
  - `n_iter` (iteraciones)
  - `bias`

---

## 🧪 Experimentos

### 🔹 1. Perceptrón sin escalado

- `eta = 0.01`
- `n_iter = 1000`
- `bias = 0.5`

- **Accuracy: 100%**

---

### 🔹 2. Perceptrón con escalado

Se utiliza:

- `RobustScaler`
- `Pipeline`
- `ColumnTransformer`
- `GridSearchCV`

Parámetros evaluados:

```python
param_grid = {
    'classifier__eta': [0.01, 0.1, 0.05],
    'classifier__n_iter': [100, 500, 1000],
    'classifier__bias': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}
```

Resultado: 

- **Accuracy: 96.67%%**
- Error: falso positivo

---

### 🔹 3. Regresión logística

Modelo:

```python
LogisticRegression(max_iter=1000)
```

- **Accuracy: 100%**

---

## 📈 Comparativa de modelos

| Modelo                     | Accuracy | Precision | Recall | F1-score |
|--------------------------|----------|----------|--------|----------|
| Perceptrón (sin escalar) | 100%     | 100%     | 100%   | 100%     |
| Perceptrón (escalado)    | 96.67%   | 96.92%   | 96.67% | 96.68%   |
| Regresión Logística      | 100%     | 100%     | 100%   | 100%     |

---

## 📌 Conclusiones

- El número de iteraciones es clave para mejorar el modelo.  
- El escalado de datos no siempre mejora el rendimiento.  
- GridSearchCV ayuda a encontrar hiperparámetros óptimos.  
- La regresión logística funciona extremadamente bien en este problema.  

---

## 🛠️ Tecnologías utilizadas

- Python  
- NumPy  
- Pandas  
- Scikit-learn  

---

## 🚀 Instalación y ejecución

```bash
pip install numpy pandas scikit-learn scipy
```

```bash
python main.py
```

---

## 📁 Estructura del proyecto

```
├── mi_modelo_AGG.py
├── main.py
├── Pass-Fail-Data.csv
└── README.md
```

---

## 👨‍💻 Autor

**Adrián García García** - [LinkedIn](https://www.linkedin.com/in/adri%C3%A1n-garc%C3%ADa-garc%C3%ADa-6ab399333/)
