# EV Range Prediction using Machine Learning

This project predicts real-world electric vehicle (EV) driving range using machine learning based on vehicle specifications and driving conditions. A Gradient Boosting Regression model is trained on a synthetic dataset and deployed using a Streamlit web application for scenario-based predictions.

## Overview

Manufacturer-claimed EV ranges often differ from real-world performance due to varying driving conditions and environmental factors. This project demonstrates how machine learning can model these relationships and estimate EV range more realistically.

## Key Highlights

- Synthetic dataset simulating EV driving scenarios
- Multiple regression models evaluated
- Gradient Boosting Regression selected using K-Fold cross-validation
- Achieved ~0.98 test R² on synthetic data
- Interactive Streamlit application for predictions

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```bash
EV-Range-Prediction/
├── EV Range Prediction.ipynb
├── app.py
├── gbr_model.pkl
├── label_encoders.pkl
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/fayiznalakath/EV-Range-Prediction.git
cd EV-Range-Prediction
pip install -r requirements.txt
streamlit run app.py
```
---

## Dataset Note
The dataset used in this project is synthetically generated to simulate realistic EV driving conditions. Model performance is intended for learning and experimentation purposes.

## Author
Fayiz N
