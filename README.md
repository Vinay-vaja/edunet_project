# 🩺 AI Disease Prediction System

### SDG 3 – Good Health and Well-being

A beginner-friendly Machine Learning project that predicts whether a patient has **Diabetes** using the Pima Indians Diabetes Dataset. The system uses a trained ML model served through a clean, professional **Streamlit** web interface.

---

## 📌 Project Description

Diabetes is one of the most widespread chronic diseases globally. Early detection can significantly improve patient outcomes and reduce healthcare costs. This project leverages **Machine Learning** to predict diabetes risk based on key health indicators, directly contributing to **UN SDG 3 (Good Health and Well-being)**.

---

## ✨ Features

- 🤖 **Dual Model Training** — Logistic Regression & Decision Tree
- 📊 **Model Comparison** — Accuracy scores, confusion matrices, classification reports
- 💾 **Model Persistence** — Best model saved using Pickle
- 🎨 **Professional Streamlit UI** — Modern gradient-based design
- 🔍 **Real-time Prediction** — Enter health data and get instant results
- 📈 **Probability Display** — Color-coded risk bars (Red = Diabetes, Green = Safe)
- 🗺️ **Correlation Heatmap** — Visual feature analysis using Seaborn
- 📂 **Dataset Explorer** — View raw data and statistical summary

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **scikit-learn** | ML model training & evaluation |
| **Streamlit** | Web application UI |
| **Matplotlib** | Plotting |
| **Seaborn** | Correlation heatmap |
| **Pickle** | Model serialization |

---

## 📁 Project Structure

```
disease_prediction/
│
├── dataset/
│   └── diabetes.csv          # Pima Indians Diabetes Dataset
│
├── train.py                  # Model training script
├── app.py                    # Streamlit web application
├── model.pkl                 # Saved best ML model
├── scaler.pkl                # Saved feature scaler
└── README.md                 # Project documentation
```

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

### Step 2: Train the Model
```bash
python train.py
```
This will:
- Load and preprocess the dataset
- Train Logistic Regression & Decision Tree models
- Compare accuracies
- Save the best model as `model.pkl`

### Step 3: Launch the Web App
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

---

## 📊 Dataset Information

**Pima Indians Diabetes Dataset** — 768 samples, 8 features

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Genetic diabetes risk function |
| Age | Age in years |
| **Outcome** | **0 = No Diabetes, 1 = Diabetes** |

---

## 🌍 SDG 3 Impact

This project aligns with **United Nations Sustainable Development Goal 3: Good Health and Well-being** by:

1. **Early Detection** — Identifying diabetes risk before symptoms appear
2. **Accessibility** — Making health screening available via a simple web interface
3. **Cost Reduction** — Enabling preliminary screening without expensive lab visits
4. **Awareness** — Educating users about key health parameters

---

## 📝 Viva Questions & Answers

### Q1: What is the purpose of this project?
**A:** This project predicts whether a patient has diabetes using Machine Learning. It uses health parameters like Glucose, BMI, and Age to make predictions, supporting early detection and SDG 3.

### Q2: Why did you choose Logistic Regression and Decision Tree?
**A:** Logistic Regression is a simple, interpretable model ideal for binary classification (Diabetes/No Diabetes). Decision Tree is easily visualizable and handles non-linear relationships. Both are beginner-friendly and effective for this problem.

### Q3: What is the role of StandardScaler?
**A:** StandardScaler normalizes features by removing the mean and scaling to unit variance. This is important because features like Glucose (0-200) and BMI (0-70) have different ranges. Without scaling, the model may give more importance to features with larger values.

### Q4: What is a Confusion Matrix?
**A:** A confusion matrix shows how many predictions were correct and incorrect. It displays True Positives, True Negatives, False Positives, and False Negatives — helping us understand not just accuracy, but the types of errors the model makes.

### Q5: How does this project relate to SDG 3?
**A:** SDG 3 aims for "Good Health and Well-being." This project supports it by using AI for early disease detection, making healthcare more accessible, and enabling data-driven health decisions.

---

## 🌐 Real-World Use Cases

### 1. Rural Health Clinics
In areas with limited access to specialists, this tool can provide preliminary diabetes screening. Health workers can input basic patient data and get instant risk assessments, enabling early referrals.

### 2. Health Insurance Risk Assessment
Insurance companies can use similar models to assess policyholder health risks, offer preventive health programs, and reduce long-term claim costs through early intervention.

---

## ⚠️ Limitations

1. **Limited Dataset** — The model is trained on only 768 samples from the Pima Indian population, which may not generalize well to other demographics.
2. **Binary Prediction** — The model only predicts Diabetes (Yes/No) and cannot assess severity or type (Type 1 vs Type 2).
3. **No Real-time Data** — The system requires manual input; it doesn't integrate with medical devices or electronic health records.
4. **Feature Constraints** — Relies on 8 specific features; important factors like diet, exercise, and family history details are not captured.
5. **Not a Medical Diagnosis** — This is a screening tool and should not replace professional medical evaluation.

---

## 📄 License

This project is for educational purposes only.

---

> Created to explore Machine Learning and impact health worldwide.
