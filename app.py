import streamlit as st
import joblib
import pandas as pd

# Charger les données depuis cleaned_data.csv
cleaned_data = pd.read_csv('cleaned_data.csv')

# Supprimer les doublons pour rome_code et rome_label
rome_data_unique = cleaned_data.drop_duplicates(subset=['rome_code', 'rome_label'], keep='first')
rome_data_unique['combined'] = rome_data_unique['rome_code'] + ' - ' + rome_data_unique['rome_label']

# Supprimer les doublons pour contract_type, experience_required, experience_required_months et job_location
contract_type_options = cleaned_data['contract_type'].drop_duplicates().tolist()
experience_required_options = cleaned_data['experience_required'].drop_duplicates().tolist()
experience_required_months_options = cleaned_data['experience_required_months'].drop_duplicates().sort_values().tolist()
job_location_options = cleaned_data['job_location'].drop_duplicates().tolist()

# Charger le modèle
model = joblib.load('salary_prediction_model.pkl')

# Titre de l'application
st.title("Prédiction de Salaire")

# Widgets pour collecter les données utilisateur avec des listes déroulantes
rome_code_selected = st.selectbox('Sélectionnez le code ROME :', rome_data_unique['combined'])
rome_code_actual = rome_code_selected.split(' - ')[0]  # Extraire le code ROME

contract_type = st.selectbox("Type de Contrat", contract_type_options)
experience_required = st.selectbox("Expérience Requise", experience_required_options)
experience_required_months = st.selectbox("Mois d'Expérience", experience_required_months_options)
job_location = st.selectbox("Lieu de Travail", job_location_options)

# Bouton pour lancer la prédiction
if st.button("Prédire le Salaire"):
    input_data = pd.DataFrame({
        'rome_code': [rome_code_actual],
        'contract_type': [contract_type],
        'experience_required': [experience_required],
        'experience_required_months': [experience_required_months],
        'job_location': [job_location]
    })

    # Faire la prédiction
    prediction = model.predict(input_data)
    st.write(f"Salaire Prévu : {prediction[0]:.2f} €")