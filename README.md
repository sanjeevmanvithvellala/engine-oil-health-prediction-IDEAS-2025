# 🚗 Engine Oil Health Prediction  
**IDEAS Internship – Autumn 2025 | ISI Kolkata**

Predicting engine oil degradation and health status using Machine Learning and Deep Learning models on sensor-derived automotive parameters such as RPM, oil temperature, pressure, and degradation indices.

Developed during the IDEAS Internship 2025 at ISI Kolkata under the mentorship of **Dr. Sandip Bhattacharyya**.

---

## 🎯 Problem Statement

Engine oil degradation directly impacts engine efficiency, fuel economy, wear rate, and maintenance cycles. Traditional oil replacement schedules are time-based and may lead to:

- Premature oil replacement (increased cost)
- Delayed replacement (engine damage risk)

This project builds a **data-driven predictive model** to classify engine oil health using real-time sensor features, enabling intelligent maintenance decisions.

---

## 👥 Team

- **Baliji Niroop**
- **Sanjeev Manvith Vellala**
- **Kottu Shyam Sailesh**
- **Vamsi Raghav Tallapaka**

---

## 🧠 Methods & Models

### Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- Gradient Boosting
- XGBoost
- Stacking Ensemble

### Deep Learning
- Keras Sequential Neural Network

### Preprocessing & Techniques
- Missing value handling
- StandardScaler (feature scaling)
- SMOTE (class imbalance handling)
- 5-fold Cross Validation
- SHAP (model explainability)

---

## 📊 Evaluation

Metrics used:
- Accuracy
- F1 Score
- ROC-AUC
- Cross-Validation Accuracy
- Confusion Matrix
- ROC Curve

Multi-model comparison implemented for evaluation transparency and reproducibility.

---

## 📂 Repository Structure

```
Data/           → Processed and raw datasets
Notebook/       → EDA, cleaning, modeling, SHAP analysis
Models/         → Saved trained models
Results/        → Evaluation metrics, confusion matrices, ROC curves
Reports/        → Internship report & slides
evaluation_report.py
model_comparison.py
```


## 🚘 Automotive Significance

From an automobile engineering perspective, this system can support:

- Predictive maintenance
- Smart oil change intervals
- Reduced engine wear
- Improved fuel efficiency
- Integration into ECU-based diagnostic systems

Future extension: Real-time IoT-based oil health monitoring.

---

## 📌 Project Repository

🔗 GitHub:  
https://github.com/sanjeevmanvithvellala/engine-oil-health-prediction-IDEAS-2025

---

## 🔒 Intellectual Property & License

This work was developed during the IDEAS Internship 2025 at ISI Kolkata under **Dr. Sandip Bhattacharyya**.

All materials are protected under Intellectual Property Rights (IPR).  
Unauthorized reuse or redistribution without citation/permission is prohibited.

© 2025 The Authors. Released under the MIT License (see LICENSE).
