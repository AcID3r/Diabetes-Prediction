"""
🏥 Diabetes Prediction System - Production App
Author: Madhav N Joshi [iitg_ds_2501358]
Dreamflow 50-Day Challenge Submission
"""

import gradio as gr
import numpy as np
import pickle
import os
from pathlib import Path

# ============================================================================
# LOAD MODELS AND SCALER
# ============================================================================

def load_models():
    """Load all trained models and scaler"""
    models = {}
    model_files = {
        'LightGBM': 'lightgbm.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'LDA': 'lda.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'SVM': 'svm.pkl',
        'KNN': 'knn.pkl'
    }
    
    # Load each model
    for name, filename in model_files.items():
        try:
            with open(f'models/{filename}', 'rb') as f:
                models[name] = pickle.load(f)
            print(f"✅ Loaded {name}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
    
    # Load scaler
    try:
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Loaded scaler")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        scaler = None
    
    # Load results (performance metrics)
    try:
        with open('models/results.pkl', 'rb') as f:
            results = pickle.load(f)
        print("✅ Loaded performance metrics")
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        # Fallback metrics
        results = {name: {'accuracy': 0.75, 'precision': 0.70, 'recall': 0.65, 'f1_score': 0.67} 
                   for name in models.keys()}
    
    return models, scaler, results

# Load everything at startup
print("🚀 Loading models...")
MODELS, SCALER, RESULTS = load_models()
print(f"✅ Loaded {len(MODELS)} models successfully!\n")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, dpf, age, model_choice):
    """
    Make diabetes prediction using selected model
    """
    
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    
    # Scale the input
    if SCALER is not None:
        input_scaled = SCALER.transform(input_data)
    else:
        input_scaled = input_data  # Fallback if scaler fails
    
    # Get selected model
    selected_model = MODELS.get(model_choice)
    
    if selected_model is None:
        return "❌ Error: Model not found. Please try another model."
    
    try:
        # Make prediction
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0][1]
    except Exception as e:
        return f"❌ Prediction Error: {str(e)}"
    
    # Get model performance metrics
    model_metrics = RESULTS.get(model_choice, {})
    model_accuracy = model_metrics.get('accuracy', 0.75)
    model_precision = model_metrics.get('precision', 0.70)
    model_recall = model_metrics.get('recall', 0.65)
    model_f1 = model_metrics.get('f1_score', 0.67)
    
    # ========================================================================
    # RISK FACTOR ANALYSIS
    # ========================================================================
    
    risk_factors = []
    
    # Glucose analysis
    if glucose > 140:
        risk_factors.append("🔴 **High Glucose Level** (>140 mg/dL) - Strongest predictor")
    elif glucose > 125:
        risk_factors.append("🟡 **Elevated Glucose** (125-140 mg/dL)")
    
    # BMI analysis
    if bmi > 30:
        risk_factors.append("🔴 **Obesity** (BMI >30) - Major risk factor")
    elif bmi > 25:
        risk_factors.append("🟡 **Overweight** (BMI 25-30)")
    
    # Age analysis
    if age > 45:
        risk_factors.append("🟡 **Age-related Risk** (>45 years)")
    
    # Genetic risk
    if dpf > 0.8:
        risk_factors.append("🟡 **High Genetic Risk** (DPF >0.8)")
    elif dpf > 0.5:
        risk_factors.append("🟡 **Moderate Genetic Risk** (DPF 0.5-0.8)")
    
    # Insulin analysis
    if insulin > 200:
        risk_factors.append("🟡 **High Insulin Level** (>200 mu U/ml)")
    
    # Blood pressure analysis
    if blood_pressure > 90:
        risk_factors.append("🔴 **High Blood Pressure** (>90 mm Hg)")
    elif blood_pressure > 80:
        risk_factors.append("🟡 **Elevated Blood Pressure** (80-90 mm Hg)")
    
    # Pregnancy analysis
    if pregnancies > 5:
        risk_factors.append("🟡 **Multiple Pregnancies** (>5)")
    
    # ========================================================================
    # GENERATE PREDICTION REPORT
    # ========================================================================
    
    is_diabetic = prediction == 1
    confidence = max(probability, 1-probability) * 100
    
    # Header
    if is_diabetic:
        header = f"""
# ⚠️ POSITIVE - High Risk for Diabetes

🔴 **Diabetes Probability:** {probability*100:.1f}%  
📊 **Confidence Level:** {confidence:.1f}%
"""
        recommendations = """
## 🏥 Immediate Medical Actions

### Priority Steps:
1. **🩺 Consult Healthcare Provider** - Schedule comprehensive diabetes screening
2. **📋 Recommended Tests:**
   - HbA1c test (3-month blood sugar average)
   - Fasting plasma glucose test
   - Oral glucose tolerance test (OGTT)

### 🥗 Lifestyle Modifications:

**Diet:**
- Reduce sugar and refined carbohydrate intake
- Increase fiber-rich foods (vegetables, whole grains, legumes)
- Follow diabetic-friendly meal plan (Mediterranean or DASH diet)
- Control portion sizes

**Exercise:**
- 30-45 minutes of moderate activity daily
- Activities: Walking, swimming, cycling, yoga
- Strength training 2-3 times per week

**Weight Management:**
- Target: 5-10% body weight reduction if overweight
- Gradual, sustainable weight loss (1-2 lbs per week)

**General Health:**
- 💧 Stay hydrated (8-10 glasses of water daily)
- 😴 Maintain 7-8 hours of quality sleep
- 🚭 Avoid smoking and limit alcohol
- 📉 Manage stress through meditation or counseling

### 📊 Monitoring Plan:
- Regular blood glucose monitoring (as advised by doctor)
- Track weight, blood pressure, and physical activity
- Follow-up appointments every 3 months
- Keep a health journal
"""
    else:
        header = f"""
# ✅ NEGATIVE - Low Risk for Diabetes

🟢 **No Diabetes Probability:** {(1-probability)*100:.1f}%  
📊 **Confidence Level:** {confidence:.1f}%
"""
        recommendations = """
## ✨ Preventive Care Recommendations

### 🥗 Maintain Healthy Lifestyle:

**Diet:**
- Continue balanced nutrition with whole grains, vegetables, fruits, lean proteins
- Limit processed foods and sugary beverages
- Practice mindful eating

**Exercise:**
- Maintain regular physical activity (150 minutes per week)
- Mix cardio and strength training
- Stay active throughout the day

**Weight Management:**
- Keep BMI in healthy range (18.5-24.9)
- Monitor weight regularly

**General Health:**
- 🚭 Avoid smoking and excessive alcohol
- 💧 Stay well-hydrated
- 😴 Prioritize quality sleep

### 📅 Regular Monitoring:
- Annual health checkups
- Blood sugar screening every 1-2 years (or as recommended)
- Monitor blood pressure and cholesterol
- Be aware of family history and genetic risks
"""
    
    # Risk factors section
    if risk_factors:
        risk_section = "## 🎯 Identified Risk Factors\n\n" + "\n".join(f"- {factor}" for factor in risk_factors)
    else:
        risk_section = "## ✅ No Major Risk Factors Identified\n\nYour health indicators are within normal ranges."
    
    # Input summary table
    input_summary = f"""
## 📋 Your Input Summary

| Parameter | Value | Status |
|-----------|-------|--------|
| **Pregnancies** | {pregnancies} | {'✅ Normal' if pregnancies < 6 else '⚠️ High'} |
| **Glucose** | {glucose} mg/dL | {'✅ Normal' if glucose < 126 else '⚠️ High'} |
| **Blood Pressure** | {blood_pressure} mm Hg | {'✅ Normal' if blood_pressure < 80 else '⚠️ Elevated' if blood_pressure < 90 else '🔴 High'} |
| **Skin Thickness** | {skin_thickness} mm | Normal |
| **Insulin** | {insulin} mu U/ml | {'✅ Normal' if insulin < 200 else '⚠️ High'} |
| **BMI** | {bmi:.1f} | {'✅ Normal' if bmi < 25 else '⚠️ Overweight' if bmi < 30 else '🔴 Obese'} |
| **Pedigree Function** | {dpf:.3f} | {'✅ Low' if dpf < 0.5 else '⚠️ Moderate' if dpf < 0.8 else '🔴 High'} |
| **Age** | {age} years | {'✅ Normal' if age < 45 else '⚠️ Elevated risk'} |
"""
    
    # Model information
    model_info = f"""
## 🤖 Model Performance

**Selected Model:** {model_choice}

| Metric | Score |
|--------|-------|
| **Accuracy** | {model_accuracy*100:.2f}% |
| **Precision** | {model_precision*100:.2f}% |
| **Recall** | {model_recall*100:.2f}% |
| **F1-Score** | {model_f1:.4f} |

**Prediction Breakdown:**
- Diabetes Probability: {probability*100:.1f}%
- No Diabetes Probability: {(1-probability)*100:.1f}%
"""
    
    # Feature importance note
    feature_note = """
## 📊 Understanding the Prediction

This AI model analyzes 8 medical parameters to predict diabetes risk:

**Most Important Factors** (based on Logistic Regression analysis):
1. **Glucose Level** (28.5%) - Primary indicator
2. **BMI** (23.1%) - Strong predictor
3. **Age** (16.8%) - Significant factor
4. **Diabetes Pedigree Function** (12.4%) - Genetic influence
5. Other factors: Pregnancies, Blood Pressure, Insulin, Skin Thickness

The model was trained on the Pima Indians Diabetes Database with 768 patients and achieved robust performance across multiple validation metrics.
"""
    
    # Medical disclaimer
    disclaimer = """
---

## ⚕️ Important Medical Disclaimer

**This is an AI-based prediction tool for educational and screening purposes.**

- ❌ **NOT a substitute** for professional medical diagnosis
- ❌ **NOT medical advice** - always consult qualified healthcare providers
- ✅ **Use as a screening tool** to identify potential risk
- ✅ **Discuss results** with your doctor for proper evaluation

**Project Information:**
- **Author:** Madhav N Joshi [iitg_ds_2501358]
- **Project:** DS Capstone - Diabetes Prediction Using Classical ML
- **Dataset:** Pima Indians Diabetes Database (768 patients, 8 features)
- **Models Tested:** Logistic Regression, LDA, Random Forest, XGBoost, LightGBM, SVM, KNN, Decision Tree

---

*This app was built as part of the Dreamflow 50-Day Challenge*
"""
    
    # Combine all sections
    full_report = f"""
{header}

{risk_section}

{recommendations}

{input_summary}

{model_info}

{feature_note}

{disclaimer}
"""
    
    return full_report

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for beautiful styling
custom_css = """
.gradio-container {
    font-family: 'Inter', 'Arial', sans-serif;
    max-width: 1400px;
    margin: auto;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 40px 20px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.metric-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 10px 0;
}

footer {
    text-align: center;
    margin-top: 50px;
    padding: 20px;
    border-top: 2px solid #e0e0e0;
}
"""

# Build the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="🏥 Diabetes Prediction System") as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1 style='margin: 0; font-size: 3rem; font-weight: 800;'>🏥 Diabetes Risk Predictor</h1>
        <p style='margin: 15px 0 5px 0; font-size: 1.3rem; opacity: 0.95;'>AI-Powered Early Detection Using Machine Learning</p>
        <p style='margin: 5px 0; font-size: 1rem; opacity: 0.9;'>Built by <strong>Madhav N Joshi</strong> [iitg_ds_2501358]</p>
        <p style='margin: 5px 0 0 0; font-size: 0.9rem; opacity: 0.85;'>Dreamflow 50-Day Challenge • DS Capstone Project</p>
    </div>
    """)
    
    gr.Markdown("""
    ## 👋 Welcome!
    
    This tool uses advanced machine learning to predict diabetes risk based on medical parameters. 
    Enter your health information below and select an AI model to get instant risk assessment.
    
    **⚡ Quick Start:** Adjust the sliders, pick a model, and click "Predict Diabetes Risk"
    """)
    
    with gr.Row():
        # Left column - Input parameters
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Patient Information")
            
            with gr.Row():
                with gr.Column():
                    pregnancies = gr.Slider(
                        minimum=0, maximum=20, value=1, step=1,
                        label="🤰 Number of Pregnancies",
                        info="Total number of pregnancies"
                    )
                    
                    glucose = gr.Slider(
                        minimum=0, maximum=200, value=120, step=1,
                        label="🩸 Glucose Level (mg/dL)",
                        info="Fasting blood sugar • Normal: <126 | Diabetes: >140"
                    )
                    
                    blood_pressure = gr.Slider(
                        minimum=0, maximum=140, value=70, step=1,
                        label="💓 Blood Pressure (mm Hg)",
                        info="Diastolic • Normal: <80 | High: >90"
                    )
                    
                    skin_thickness = gr.Slider(
                        minimum=0, maximum=100, value=20, step=1,
                        label="📏 Skin Thickness (mm)",
                        info="Triceps skin fold measurement"
                    )
                
                with gr.Column():
                    insulin = gr.Slider(
                        minimum=0, maximum=900, value=80, step=1,
                        label="💉 Insulin Level (mu U/ml)",
                        info="2-hour serum insulin • Normal: <200"
                    )
                    
                    bmi = gr.Slider(
                        minimum=10.0, maximum=70.0, value=25.0, step=0.1,
                        label="⚖️ BMI (Body Mass Index)",
                        info="Normal: <25 | Overweight: 25-30 | Obese: >30"
                    )
                    
                    dpf = gr.Slider(
                        minimum=0.0, maximum=2.5, value=0.5, step=0.01,
                        label="🧬 Diabetes Pedigree Function",
                        info="Genetic influence • Low: <0.5 | High: >0.8"
                    )
                    
                    age = gr.Slider(
                        minimum=18, maximum=100, value=30, step=1,
                        label="🎂 Age (years)",
                        info="Risk increases significantly after 45"
                    )
        
        # Right column - Model selection
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 AI Model Selection")
            
            model_choice = gr.Radio(
                choices=[
                    "LightGBM",
                    "Random Forest",
                    "XGBoost",
                    "Logistic Regression",
                    "LDA",
                    "Decision Tree",
                    "SVM",
                    "KNN"
                ],
                value="LightGBM",
                label="Choose Prediction Model",
                info="Each model has unique strengths"
            )
            
            # Model performance table
            perf_table = "| Model | Accuracy | F1-Score |\n|-------|----------|----------|\n"
            for model_name in ["LightGBM", "Random Forest", "XGBoost", "Logistic Regression", "LDA"]:
                metrics = RESULTS.get(model_name, {})
                acc = metrics.get('accuracy', 0.75) * 100
                f1 = metrics.get('f1_score', 0.67)
                perf_table += f"| **{model_name}** | {acc:.1f}% | {f1:.4f} |\n"
            
            gr.Markdown(f"""
            ### 📊 Model Performance
            
            {perf_table}
            
            **💡 Recommendation:**  
            **LightGBM** offers the best balance of accuracy and recall, making it ideal for medical screening.
            
            **Logistic Regression** provides interpretable results with clear feature importance.
            """)
    
    # Predict button
    gr.Markdown("---")
    predict_btn = gr.Button(
        "🔬 Predict Diabetes Risk",
        variant="primary",
        size="lg",
        scale=2
    )
    
    # Output section
    gr.Markdown("---")
    gr.Markdown("## 📊 Prediction Results")
    output = gr.Markdown(label="Results will appear here")
    
    # Connect prediction function
    predict_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age, model_choice],
        outputs=output
    )
    
    # Example cases
    gr.Markdown("---")
    gr.Markdown("## 🧪 Try Example Cases")
    gr.Markdown("Click any example below to auto-fill the form and see instant predictions")
    
    gr.Examples(
        examples=[
            [6, 148, 72, 35, 0, 33.6, 0.627, 50, "LightGBM"],  # High risk
            [1, 85, 66, 29, 0, 26.6, 0.351, 31, "LightGBM"],   # Low risk
            [8, 183, 64, 0, 0, 23.3, 0.672, 32, "LightGBM"],   # Borderline
            [0, 95, 70, 25, 100, 22.0, 0.200, 25, "LightGBM"], # Very low risk
            [10, 168, 74, 0, 0, 38.0, 0.537, 60, "LightGBM"],  # Very high risk
        ],
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age, model_choice],
        label="Example Patients",
        examples_per_page=5
    )
    
    # Footer
    gr.HTML("""
    <footer>
        <h3>🎓 About This Project</h3>
        <p><strong>Project:</strong> DS Capstone - Diabetes Prediction Using Classical Machine Learning</p>
        <p><strong>Author:</strong> Madhav N Joshi [iitg_ds_2501358]</p>
        <p><strong>Dataset:</strong> Pima Indians Diabetes Database (768 patients, 8 features)</p>
        <p><strong>Models:</strong> 8 ML algorithms trained and compared</p>
        <p><strong>Best Model:</strong> LightGBM (75.97% accuracy, 0.6542 F1-score)</p>
        <p style="margin-top: 20px; font-size: 0.9em; color: #666;">
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational and screening purposes only. 
            Not a substitute for professional medical diagnosis. Always consult healthcare providers.
        </p>
        <p style="margin-top: 15px;">
            🚀 Built for the <strong>Dreamflow 50-Day Challenge</strong>
        </p>
    </footer>
    """)

# ============================================================================
# LAUNCH THE APP
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Diabetes Prediction App...")
    print(f"📊 Loaded {len(MODELS)} models")
    print("🌐 Launching Gradio interface...\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )