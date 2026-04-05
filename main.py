from mi_modelo_AGG import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="This Pipeline instance is not fitted yet"
)


def check_succes_mistakes(y_test, predicts):
    success = 0
    mistakes = 0
    zeroSuccess=0
    zeroMistakes=0
    oneSuccess=0
    oneMistakes=0
    for i, j in zip(y_test, predicts):
        if i==j:
            success+=1
            if i==1:
                oneSuccess+=1
            else:
                zeroSuccess+=1
        else:
            mistakes+=1
            if j==1 and i==0:
                zeroMistakes+=1 # Es decir, cuando en realidad debería haber sido uno pero ha dado 0
            else:
                oneMistakes+=1
    return success, mistakes, zeroSuccess, zeroMistakes, oneSuccess, oneMistakes


# Sin escalar
studentsDf = pd.read_csv("Pass-Fail-Data.csv", sep=",")
perceptron = Perceptron(0.01, 1000)
x = studentsDf[['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week']]
y = studentsDf['pass']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
perceptron.fit(X_train, y_train)
predicts = perceptron.predict(X_test)
acc_m1 = accuracy_score(y_test, predicts)*100
rep_p1 = classification_report(y_test, predicts, output_dict=True)

# Creamos estas variables para almacenar el número de acierto, errores
# El número de aciertos cuando ha salido 1, el número de aciertos cuando ha salido 0
# y lo mismo pero con los errores
success, mistakes, zeroSuccess, zeroMistakes, oneSuccess, oneMistakes = check_succes_mistakes(y_test, predicts)
print(f"{round(acc_m1,2)}% success rate")
print(f"It has made a total of {success} correct answers and has had {mistakes} errors")
print(f"PASSED (true positive): {oneSuccess}")
print(f"PASSED (false positive): {oneMistakes}")
print(f"FAIL (true negative): {zeroSuccess}")
print(f"FAIL (false negative): {zeroMistakes}")

# -------------------------------------------

# Con escalamiento de datos
numeric_features = ["attendance_pct" , "homework_pct","midterm_score","study_hours_per_week"]
perceptronScaledData = Perceptron(eta=0.01, n_iter=1000)
robustScaler = RobustScaler()
numeric_transformer = Pipeline(steps=[ 
    ('scaler', RobustScaler())                     # Escala a media=0, std=1
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='drop'  # Columnas no especificadas se eliminan
)
pipeline_complete = Pipeline(steps=[
    ('preprocessor', preprocessor),                              # Preprocesa
    ('classifier', Perceptron(eta=0.01, n_iter=1000))
])

param_grid = {
    
    # Hiperparámetros del Perceptrón
    'classifier__eta': [0.01, 0.1, 0.05],
    'classifier__n_iter': [100, 500, 1000],
    'classifier__bias': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}

grid_search = GridSearchCV(
    pipeline_complete,
    param_grid,
    cv=3,                # Validación cruzada con 3 particiones
    scoring='accuracy',  # Métrica a optimizar
    n_jobs=1,
    verbose=1            # Mostrar progreso
)

# Esta parte hace que tarde unos cuantos segundos (10-20 aprox) en ejecutarse, ya que está
# probando todas las combinaciones posibles pasadas como parámetro

# Ejecutamos la búsqueda
print("In this case, we trained the model using GridSearch to find the best combination of parameters")
print()
grid_search.fit(X_train, y_train)

# Resultados
print("Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

print()
print(f"Best score in cross-validation: {grid_search.best_score_:.4f}")

# Evaluamos el mejor modelo en el conjunto de test
best_model = grid_search.best_estimator_  # El mejor pipeline
y_pred_best = best_model.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best) * 100
rep_p2 = classification_report(y_test, y_pred_best, output_dict=True)
success, mistakes, zeroSuccess, zeroMistakes, oneSuccess, oneMistakes = check_succes_mistakes(y_test, y_pred_best)

print(f"{round(acc_best,2)}% succes rate")
print(f"It has made a total of {success} correct answers and has had {mistakes} errors")
print(f"PASSED (true positive): {oneSuccess}")
print(f"PASSED (false positive): {oneMistakes}")
print(f"FAIL (true negative): {zeroSuccess}")
print(f"FAIL (false negative): {zeroMistakes}")

# -------------------------------------------------------------------------

# AMPLIACIÓN
# Comparación con un modelo de clasificación binaria, Regresión Logística
# Como ya tenemos los datos x e y, vamos a separarlo para obtener datos de entrenamiento y de test

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
# Para ser lo más justos posible, indicaremos parámetros similares que la neurona creada anteriormente
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
logistic_regression.fit(X_train, y_train)
predicts = logistic_regression.predict(X_test)
acc_m3 = accuracy_score(y_test, predicts)*100
rep_lr = classification_report(y_test, predicts, output_dict=True)
# Vemos que obtiene un 100% de tasa de acierto
print(f"Logistic regression accuracy rate: {round(acc_m3,2)}%")
print()
# Por último, haremos una comparativa entre los tres modelos

comparative_table = pd.DataFrame({
    "Model": [
        "Non-scaling perceptron",
        "Perceptron with scaling",
        "Logistic Regression"
    ],
    "Accuracy": [
        acc_m1,
        acc_best,
        acc_m3
    ],
    "Precision": [
        rep_p1["weighted avg"]["precision"],
        rep_p2["weighted avg"]["precision"],
        rep_lr["weighted avg"]["precision"]
    ],
    "Recall": [
        rep_p1["weighted avg"]["recall"],
        rep_p2["weighted avg"]["recall"],
        rep_lr["weighted avg"]["recall"]
    ],
    "F1-score": [
        rep_p1["weighted avg"]["f1-score"],
        rep_p2["weighted avg"]["f1-score"],
        rep_lr["weighted avg"]["f1-score"]
    ]
})

print("The following report compares the three models developed in this activity:")
print(comparative_table)