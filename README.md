# 🩺 Disease Prediction & Medical Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)

---

## 📌 Table of Contents

* [🧠 Project Overview](#-project-overview)
* [📁 Project Structure](#-project-structure)
* [🚀 How to Run](#-how-to-run)
* [🔍 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
* [🧠 Why Random Forest?](#-why-random-forest)
* [🔁 ML Model Comparison](#-ml-model-comparison)
* [🧪 Future Improvements](#-future-improvements)
* [⚠️ Known Issues & Errors](#️-known-issues--errors)
* [🎥 Demo Instructions](#-demo-instructions)
* [📬 Contact](#-contact)

---

## 🧠 Project Overview

This project predicts a probable disease based on user-selected symptoms and offers corresponding:

* 📘 Disease description
* 🛡️ Preventive precautions
* 💊 Recommended medications
* 🥗 Suggested dietary changes

Built using a trained **Random Forest model**, this tool can assist in quick, non-clinical pre-diagnosis scenarios.

---

## 📁 Project Structure

```
Disease-Prediction-with-Medical-Recommendation/
│
├── data/
│   ├── description.csv
│   ├── diets.csv
│   ├── medications.csv
│   ├── precautions_df.csv
│   ├── symptoms_df.csv
│   ├── symptom_severity.csv
│   └── Training.csv
│
├── model/
│   ├── disease_prediction_model.pkl
│   └── label_encoder.pkl
│
├── app.py               # Streamlit frontend
├── disease_model.ipynb  # Model training + EDA + experiments
└── README.md
```

---

## 🚀 How to Run

### 🔧 Requirements

Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib xgboost
```

### ▶️ Launch the App

```bash
cd Disease-Prediction-with-Medical-Recommendation
streamlit run app.py
```

> Open browser at [http://localhost:8501](http://localhost:8501)

---

## 🔍 Exploratory Data Analysis (EDA)

### 📊 Disease Distribution

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(y=training['prognosis'], order=training['prognosis'].value_counts().index)
plt.title("Frequency of Diseases in Dataset")
```

### 🔥 Symptom Importance

```python
importances = model.feature_importances_
symptom_importance = pd.Series(importances, index=symptom_list).sort_values(ascending=False)
symptom_importance.head(10).plot(kind='barh')
plt.title("Top Predictive Symptoms")
```

---

## 🧠 Why Random Forest?

The **Random Forest Classifier** was selected for its:

* ✅ Robustness against overfitting
* ✅ Native support for multi-class classification
* ✅ Excellent performance with sparse, binary features (0/1 for symptoms)
* ✅ Built-in feature importance for model explainability
* ✅ Minimal data preprocessing required

---

## 🔁 ML Model Comparison

| Model                   | Pros                                                            | Cons                                                                |
| ----------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Random Forest**       | High accuracy, less overfitting, interpretable, works well OOTB | Slower on massive datasets, less fine-tuned than XGBoost            |
| **XGBoost**             | Boosted trees = better performance on large, complex data       | Sensitive to overfitting, requires hyperparameter tuning            |
| **Logistic Regression** | Fast, interpretable, easy to implement                          | Poor with nonlinear relationships, assumes linear decision boundary |
| **SVM**                 | Powerful for binary and small datasets                          | Not ideal for multi-class, slow training time on large data         |
| **KNN**                 | No training cost, intuitive                                     | Poor scalability, sensitive to irrelevant features/noise            |

---

## 🧪 Future Improvements

* [ ] Predict top 3 likely diseases with confidence scores
* [ ] SHAP-based explainability of symptoms
* [ ] Docker support for easier deployment
* [ ] PDF report generator with predictions & recommendations
* [ ] Authentication and history tracking for clinical use

---

## ⚠️ Known Issues & Errors

### ⚠️ 1. Irrelevant Symptoms → Incorrect Predictions

If the user selects symptoms that:

* Do not co-occur in any disease pattern
* Are out of context

> The model may return an inaccurate prediction or fail to match with training patterns.

🛠️ **Solution:** Added logic to warn users when the selected symptoms don’t match any known profile.

---

### ⚠️ 2. Only One Symptom → Unreliable Result

If only **one symptom** is selected:

> The prediction may be vague or inaccurate due to lack of specificity.

🛠️ **Solution:** UI prompts the user to select **at least two** symptoms for better accuracy.

---

### ⚠️ 3. UserWarning from `scikit-learn`:

```
UserWarning: X does not have valid feature names...
```

**Reason:** Input vector is a NumPy array, not a DataFrame.

🛠️ **Fix:** Wrap input in `pd.DataFrame([input_vector], columns=symptom_list)` before passing to `.predict()`.

---

## 🎥 Demo Instructions

### Run from Command Prompt

```bash
cd C:\Users\venka\Desktop\Short Projects\Disease-Prediction-with-Medical-Recommendation
python -m streamlit run "app.py"
```

Browser opens automatically at `http://localhost:8501`
✅ App is fully functional with dropdown symptom selection and multiple recommendation panels.

---

## 📬 Contact

**Author:** Venkata Vivek Varma Alluru
📧 Email: [a.v.vivekvarma@gmail.com](mailto:a.v.vivekvarma@gmail.com)
🔗 GitHub: [Avvv19](https://github.com/Avvv19)
🔗 LinkedIn: [venkatavivekvarmaalluru](https://www.linkedin.com/in/venkatavivekvarmaalluru/)



