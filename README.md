# 🏥 Diabetes Risk Predictor

**AI-Powered Early Detection Using Machine Learning**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/diabetes-predictor)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This web application uses advanced machine learning algorithms to predict diabetes risk based on medical parameters. Built as part of the **Dreamflow 50-Day Challenge** and DS Capstone Project.

**Live Demo:** [Try it here!](https://huggingface.co/spaces/YOUR_USERNAME/diabetes-predictor)

## ✨ Features

- 🤖 **8 ML Models** - Choose from LightGBM, Random Forest, XGBoost, Logistic Regression, LDA, Decision Tree, SVM, and KNN
- 📊 **Real-time Predictions** - Instant diabetes risk assessment
- 🎯 **Risk Factor Analysis** - Identifies specific health concerns
- 📈 **Model Performance Metrics** - Transparent accuracy, precision, recall, and F1-scores
- 💡 **Medical Recommendations** - Personalized health advice based on results
- 🌐 **User-Friendly Interface** - Beautiful, intuitive design built with Gradio

## 🚀 Quick Start

### Option 1: Use Online (Recommended)
Visit the live app: [https://huggingface.co/spaces/YOUR_USERNAME/diabetes-predictor](https://huggingface.co/spaces/YOUR_USERNAME/diabetes-predictor)

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/diabetes-predictor
cd diabetes-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open your browser to `http://localhost:7860`

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LightGBM** | 75.97% | 66.67% | 64.81% | 0.6542 |
| **Random Forest** | 75.97% | 68.09% | 62.96% | 0.6538 |
| **XGBoost** | 75.32% | 63.83% | 62.96% | 0.6339 |
| **Logistic Regression** | 77.92% | 73.33% | 40.74% | 0.5238 |
| **LDA** | 77.92% | 73.33% | 40.74% | 0.5238 |

**Key Insight:** LightGBM achieves the best recall (64.81%), making it ideal for medical screening where detecting diabetes cases is critical.

## 🧬 Dataset

- **Source:** Pima Indians Diabetes Database (UCI Machine Learning Repository)
- **Samples:** 768 patients
- **Features:** 8 medical parameters
- **Target:** Binary classification (Diabetes: Yes/No)

### Features:
1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration (mg/dL)
3. **Blood Pressure** - Diastolic blood pressure (mm Hg)
4. **Skin Thickness** - Triceps skin fold thickness (mm)
5. **Insulin** - 2-Hour serum insulin (mu U/ml)
6. **BMI** - Body mass index (weight in kg/(height in m)^2)
7. **Diabetes Pedigree Function** - Genetic influence score
8. **Age** - Age in years

## 🛠️ Technology Stack

- **Frontend:** Gradio 4.44.0
- **ML Framework:** Scikit-learn 1.3.0
- **Gradient Boosting:** XGBoost 2.0.3, LightGBM 4.1.0
- **Data Processing:** NumPy 1.24.3, Pandas 2.0.3
- **Visualization:** Matplotlib 3.7.2, Seaborn 0.12.2

## 📁 Project Structure

```
diabetes-predictor/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── models/               # Trained model files
    ├── lightgbm.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── logistic_regression.pkl
    ├── lda.pkl
    ├── decision_tree.pkl
    ├── svm.pkl
    ├── knn.pkl
    ├── scaler.pkl
    └── results.pkl
```

## 🎓 Academic Context

**Project:** DS Capstone - Diabetes Prediction Using Classical ML Approaches  
**Author:** Madhav N Joshi [iitg_ds_2501358]  
**Institution:** [Your Institution]  
**Date:** January 2026

### Research Highlights:
- ✅ Compared 8 classical ML algorithms
- ✅ Achieved 75.97% test accuracy with ensemble methods
- ✅ Identified Glucose (28.5%) and BMI (23.1%) as top predictors
- ✅ Demonstrated practical medical AI application

## 🏆 Dreamflow 50-Day Challenge

This project was built as part of the **Dreamflow 50-Day Challenge** (November 17, 2025 - January 25, 2026).

**Category:** Open Innovation  
**Goal:** Build and publish a real-world AI application  
**Achievement:** Production-ready diabetes prediction tool deployed to the web

## ⚕️ Medical Disclaimer

**IMPORTANT:** This tool is for educational and screening purposes only.

- ❌ NOT a substitute for professional medical diagnosis
- ❌ NOT medical advice
- ✅ Use as an initial screening tool
- ✅ Always consult qualified healthcare providers for proper diagnosis

## 📝 License



## 🙏 Acknowledgments

- **Dataset:** Pima Indians Diabetes Database (UCI ML Repository)
- **Frameworks:** Scikit-learn, XGBoost, LightGBM teams
- **Platform:** Hugging Face Spaces for hosting
- **Challenge:** Dreamflow for the 50-Day Challenge opportunity

## 📧 Contact

**Madhav N Joshi**  
Student ID: iitg_ds_2501358  
Email: [Your Email]  
GitHub: [Your GitHub]

---

**Built with ❤️ for the Dreamflow 50-Day Challenge**