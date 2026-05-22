<div align="center">

# Disease Prediction with Medical Recommendation

### AI-Powered Disease Diagnosis and Personalized Medical Recommendations

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

</div>

---

## Overview

An end-to-end machine learning system that predicts potential diseases based on patient symptoms and provides personalized medical recommendations. This project combines multiple ML algorithms with a clinical knowledge base to deliver accurate diagnoses and actionable health recommendations — bridging the gap between AI and practical healthcare delivery.

---

## Key Features

- **Multi-Model Prediction** — Random Forest, SVM, Naive Bayes, and ensemble methods
- **Symptom-Based Diagnosis** — Predicts disease from 100+ symptom inputs
- **Medical Recommendations** — Personalized medication, diet, workout, and precaution advice
- **Disease Frequency Analysis** — Visualizes disease distribution in the dataset
- **Symptom Correlation Heatmap** — Identifies patterns between symptoms and diseases
- **Interactive Web App** — Streamlit-based UI for real-time predictions
- **Explainable AI** — Feature importance to understand which symptoms drive predictions

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## Models Used

| Model | Purpose | Accuracy |
|-------|---------|----------|
| Random Forest | Primary disease classifier | ~95% |
| Support Vector Machine | Secondary classifier | ~93% |
| Naive Bayes | Probabilistic prediction | ~90% |
| Ensemble (Voting) | Final prediction | ~96% |

---

## Dataset

- **Symptoms Dataset** — 132 symptoms mapped to diseases
- **Disease Information** — Descriptions, precautions, medications, diet, and workouts
- **Training Split** — 80/20 train-test split with cross-validation

---

## Project Structure

```
Disease-Prediction/
|-- disease_model.ipynb     # Model training and evaluation
|-- disease_model.py        # Model as Python script
|-- app.py                  # Streamlit web application
|-- Disease Frequency.jpg   # Disease distribution visualization
|-- Symptom Correlation Heatmap.jpg  # Symptoms correlation analysis
|-- Project structure.txt   # Architecture overview
|-- README.md
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Avvv19/Disease-Prediction-with-Medical-Recommendation-.git
cd Disease-Prediction-with-Medical-Recommendation-

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py

# Or run the Jupyter notebook
jupyter notebook disease_model.ipynb
```

---

## Sample Output

Enter symptoms -> Get disease prediction + recommendations:
- **Predicted Disease:** Diabetes
- **Medication Suggestions:** Metformin, Glipizide
- **Diet Recommendations:** Low-sugar diet, high fiber foods
- **Workout Advice:** 30 min cardio daily
- **Precautions:** Monitor blood sugar, stay hydrated

---

## Author

**Venkata Vivek Varma Alluru** | AI in Healthcare

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/venkatavivekvarmaalluru/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Avvv19)
