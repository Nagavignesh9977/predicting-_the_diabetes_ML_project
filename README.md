Diabetes Prediction using SVM

This project presents a machine learning approach to predict the likelihood of diabetes in patients using Support Vector Machine (SVM). It includes data preprocessing with StandardScaler, model training, evaluation, and prediction.

Project Overview

Diabetes is a chronic disease that affects millions worldwide. Early detection can significantly improve treatment outcomes. This project builds a binary classification model to predict diabetes based on patient health metrics.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (optional for visualization)

Dataset

The dataset used is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) from Kaggle. It contains medical diagnostic measurements for female patients of Pima Indian heritage.

- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: `Outcome` (0 = non-diabetic, 1 = diabetic)
Project Workflow

1. Data Loading: Load the dataset using pandas.
2. Preprocessing:
   - Handle missing or zero values.
   - Normalize features using `StandardScaler`.
3. Train-Test Split: Split the data into training and testing sets.
4. Model Training: Train an SVM classifier on the scaled training data.
5. Evaluation:
   - Accuracy
   - Confusion Matrix
     
6. Prediction: Predict diabetes on test data and evaluate performance.

## 📈 Results

The SVM model achieved the following metrics (example values):

- Accuracy: 77.5%
- Precision: 74.2%
- Recall: 70.1%
- F1-score: 72.1%
- ROC-AUC: 0.80

> Note: Results may vary depending on preprocessing and hyperparameter tuning.

