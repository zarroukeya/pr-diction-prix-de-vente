from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pyodbc
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Connecter à la base de données
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-RSFN8HH\SQLEXPRESS;DATABASE=immobilier;UID=sa;PWD=12356')
query = "SELECT * FROM Dimmm_immobilier"
df = pd.read_sql(query, conn)

# Division des données en caractéristiques (X) et variable cible (y)
X = df[['budget_construction_reel']]  # Caractéristiques
y = df['prix_de_vente']  # Variable cible

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Définition de l'API
app = FastAPI()

# Middleware CORS pour autoriser les requêtes de n'importe quel domaine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Définition du schéma de la requête
class InputData(BaseModel):
    budget_construction_reel: float

# Définition de la route pour la prédiction
@app.post("/predict/")
async def predict_price(data: InputData):
    budget = data.budget_construction_reel
    
    # Effectuer la prédiction
    predicted_price = model.predict(np.array([[budget]]))[0]
    
    return {"predicted_price": predicted_price}
