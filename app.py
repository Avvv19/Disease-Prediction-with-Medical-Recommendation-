import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast  # For safely evaluating string lists

# Load model and label encoder
model = joblib.load(r'C:\Users\venka\Desktop\Short Projects\Disease-Prediction-with-Medical-Recommendation\model\disease_prediction_model.pkl')
le = joblib.load(r'C:\Users\venka\Desktop\Short Projects\Disease-Prediction-with-Medical-Recommendation\model\label_encoder.pkl')

# Load CSVs
description = pd.read_csv(r'C:\Users\venka\Desktop\Short Projects\Disease-Prediction-with-Medical-Recommendation\data\description.csv')
diets = pd.read_csv(r'C:\Users\venka\Desktop\Short Projects\Disease-Prediction-with-Medical-Recommendation\data\diets.csv')
medications = pd.read_csv(r'C:\Users\venka\Desktop\Short Projects\Disease-Prediction-with-Medical-Recommendation\data\medications.csv')
precautions_df = pd.read_csv(r'C:\Users\venka\Desktop\Short Projects\Disease-Prediction-with-Medical-Recommendation\data\precautions_df.csv')
training = pd.read_csv(r'C:\Users\venka\Desktop\Short Projects\Disease-Prediction-with-Medical-Recommendation\data\Training.csv')

# Use the same symptom list used in training data
symptom_list = training.columns.drop('prognosis').tolist()

# Streamlit UI setup
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction & Medical Recommendation")
st.markdown("Select your symptoms from the list below:")

selected_symptoms = st.multiselect("🧾 Symptoms", sorted(symptom_list))

def validate_symptoms(symptoms):
    return [sym for sym in symptoms if sym in symptom_list]

if st.button("🔍 Predict Disease"):
    if len(selected_symptoms) == 0:
        st.error("❗ Please select at least one symptom.")
    else:
        valid_symptoms = validate_symptoms(selected_symptoms)

        if len(valid_symptoms) == 0:
            st.warning("⚠️ None of the entered symptoms are recognized. Please check the symptoms and try again.")
        elif len(valid_symptoms) < 2:
            st.info("ℹ️ Please select at least two valid symptoms for a more accurate diagnosis.")
        else:
            input_vector = np.zeros(len(symptom_list))
            for symptom in valid_symptoms:
                index = symptom_list.index(symptom)
                input_vector[index] = 1

            # Check if such a symptom combination exists in training data
            training_binary = training.drop(columns=['prognosis'])
            mask = (training_binary[valid_symptoms] == 1).all(axis=1)
            if not mask.any():
                st.warning("⚠️ The selected symptoms do not match any known condition in our database.")
                st.info("💡 Please consult a medical professional for an accurate diagnosis.")
            else:
                try:
                    prediction_encoded = model.predict([input_vector])[0]
                    prediction = le.inverse_transform([prediction_encoded])[0]
                    normalized_prediction = prediction.strip().lower()

                    st.success(f"✅ Predicted Disease: **{prediction}**")

                    # 📘 Description
                    desc = description[description['Disease'].str.strip().str.lower() == normalized_prediction]['Description'].values
                    if len(desc):
                        st.subheader("📘 Description")
                        st.write(desc[0])

                    # 🛡️ Precautions
                    prec_rows = precautions_df[precautions_df['Disease'].str.strip().str.lower() == normalized_prediction]
                    if not prec_rows.empty:
                        st.subheader("🛡️ Precautions")
                        for i in range(1, 5):
                            col = f'Precaution_{i}'
                            if col in prec_rows.columns:
                                precautions = prec_rows[col].dropna().values
                                for p in precautions:
                                    st.write(f"🔸 {p}")

                    # 💊 Medications
                    meds = medications[medications['Disease'].str.strip().str.lower() == normalized_prediction]['Medication'].dropna().values
                    if len(meds):
                        st.subheader("💊 Medications")
                        for med_entry in meds:
                            if isinstance(med_entry, str) and med_entry.startswith('[') and med_entry.endswith(']'):
                                med_list = ast.literal_eval(med_entry)
                                for m in med_list:
                                    st.write(f"💊 {m}")
                            else:
                                st.write(f"💊 {med_entry}")

                    # 🥗 Diet Recommendations
                    diet = diets[diets['Disease'].str.strip().str.lower() == normalized_prediction]['Diet'].dropna().values
                    if len(diet):
                        st.subheader("🥗 Diet Recommendations")
                        for diet_entry in diet:
                            if isinstance(diet_entry, str) and diet_entry.startswith('[') and diet_entry.endswith(']'):
                                diet_list = ast.literal_eval(diet_entry)
                                for d in diet_list:
                                    st.write(f"🥗 {d}")
                            else:
                                st.write(f"🥗 {diet_entry}")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
