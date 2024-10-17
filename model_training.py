from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

# Chargement des données
cleaned_data = pd.read_csv('cleaned_data.csv')

# Suppression des colonnes redondantes ou non pertinentes
columns_to_drop = ['job_id', 'job_title', 'rome_label', 'contract_label', 'experience_label', 'contract_nature']
cleaned_data = cleaned_data.drop(columns=columns_to_drop)

# Préparation des données
X_new = cleaned_data[['rome_code', 'contract_type', 'experience_required', 'experience_required_months', 'job_location']]
y_new = cleaned_data['calculated_salary']

# Division des données en ensemble d'entraînement et de test (70% train, 30% test)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.3, random_state=42)

# Prétraitement avec normalisation
categorical_features_new = ['rome_code', 'contract_type', 'experience_required', 'job_location']
numeric_features_new = ['experience_required_months']

preprocessor_new = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features_new),  # Normalisation des variables numériques
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_new)  # Encodage des variables catégorielles
])

# Création du modèle Random Forest avec encodage et normalisation
rf_model_new = Pipeline(steps=[('preprocessor', preprocessor_new), 
                               ('regressor', RandomForestRegressor(n_estimators=90, max_depth=95, min_samples_split=3, random_state=42))])

# Entraînement du modèle
rf_model_new.fit(X_train_new, y_train_new)

# Sauvegarder le modèle entraîné
joblib.dump(rf_model_new, 'salary_prediction_model.pkl')

print("Modèle sauvegardé sous 'salary_prediction_model.pkl'")