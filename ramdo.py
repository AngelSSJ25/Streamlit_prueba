import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Cargar dataset
data = pd.read_csv("loan.csv")

# Eliminar columnas irrelevantes
if "Loan_ID" in data.columns:
    data = data.drop(columns=["Loan_ID"])

# Limpiar columnas numéricas
numeric_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]
for col in numeric_cols:
    data[col] = data[col].astype(str).str.replace(".", "", regex=False)
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Variable objetivo
y = data["Loan_Status"].map({"Y":1,"N":0})

# Codificación manual (para que coincida con la app)
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 2})
data["Married"] = data["Married"].map({"No": 1, "Yes": 2})
data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate": 0})
data["Self_Employed"] = data["Self_Employed"].map({"No": 1, "Yes": 2})

# One Hot Encoding para Dependents y Property_Area
dependents_dummies = pd.get_dummies(data["Dependents"], prefix="Dependents")
property_dummies = pd.get_dummies(data["Property_Area"], prefix="Property_Area")
data = pd.concat([data, dependents_dummies, property_dummies], axis=1)

# Selección de columnas en el orden correcto
X = data[[
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Dependents_0",
    "Dependents_1",
    "Dependents_2",
    "Dependents_3+",
    "Property_Area_Rural",
    "Property_Area_Urban",
    "Property_Area_Semiurban"
]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train,y_train)

# Guardar modelo
pickle.dump(model, open("Random_Forest.sav","wb"))

# Validación
y_pred = model.predict(X_test)
print("Accuracy:", model.score(X_test,y_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
