# 🌙 Sleep Quality Predictor

A machine learning web app that predicts your sleep quality based on lifestyle and sleep data — with smartwatch integration support.

---

## 📌 Overview

Sleep Quality Predictor uses an **XGBoost classifier** trained on real sleep data to predict whether your sleep quality is **Poor**, **Fair**, or **Good** — and gives you personalized suggestions to improve it.

Users without a smartwatch get a friendly experience using healthy average sleep stage values, while smartwatch users (Fitbit, Apple Watch, Garmin) can input their exact REM, Deep, and Light sleep percentages for higher accuracy.

---

## ✨ Features

- 🔍 **Sleep quality prediction** — Poor / Fair / Good classification
- ⌚ **Smartwatch toggle** — Enter exact sleep stages or use healthy averages
- 📊 **Confidence chart** — See how confident the model is in its prediction
- 🧠 **Sleep stage breakdown** — Pie chart of your REM / Deep / Light sleep
- 📈 **Your sleep vs ideal** — Bar chart comparing your metrics to healthy targets
- 💡 **Personalized suggestions** — Actionable tips based on your inputs
- 🏅 **Sleep score** — A 0–100 score based on model confidence

---

## 🗂️ Project Structure

```
sleep-quality-predictor/
├── data/
│   ├── raw/               ← Raw dataset (not tracked by git)
│   └── processed/         ← Cleaned & processed data (not tracked)
├── models/
│   ├── sleep_model.pkl    ← Trained XGBoost model
│   ├── scaler.pkl         ← StandardScaler
│   ├── confusion_matrix_xgb.png
│   └── feature_importance.png
├── src/
│   ├── preprocess.py      ← Data cleaning
│   ├── features.py        ← Feature engineering
│   ├── train.py           ← Model training
│   ├── predict.py         ← CLI prediction
│   └── app.py             ← Streamlit dashboard
└── requirements.txt
```

---

## 📊 Dataset

**Sleep Efficiency Dataset** from Kaggle  
👉 [https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency](https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency)

Contains sleep records with features including age, sleep duration, REM/Deep/Light sleep percentages, awakenings, caffeine, alcohol, exercise, and smoking status.

---

## 🤖 Model

| Model | Accuracy |
|-------|----------|
| Random Forest (baseline) | 88% |
| **XGBoost + GridSearchCV** | **90%** ✅ |

**Features used:**
- Age
- Sleep duration
- REM sleep %
- Deep sleep %
- Light sleep %
- Awakenings
- Caffeine consumption
- Alcohol consumption
- Exercise frequency
- Smoking status

**Target classes:**
- 🔴 Poor (sleep efficiency < 70%)
- 🟡 Fair (sleep efficiency 70–85%)
- ✅ Good (sleep efficiency > 85%)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/pabi1234810/Sleep_Pattern.git
cd Sleep_Pattern/sleep-quality-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency) and place `Sleep_Efficiency.csv` in `data/raw/`.

### 5. Run the pipeline
```bash
python src/preprocess.py
python src/features.py
python src/train.py
```

### 6. Launch the dashboard
```bash
streamlit run src/app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas / NumPy | Data processing |
| Scikit-learn | Preprocessing & evaluation |
| XGBoost | ML classifier |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Web dashboard |
| Git / GitHub | Version control |

---

## 🌿 Branches

| Branch | Description |
|--------|-------------|
| `main` | Stable version with XGBoost (90%) |
| `xgboost-model` | XGBoost experimentation branch |

---

## 💡 Future Improvements

- [ ] Connect directly to Fitbit / Apple Health API
- [ ] Add LSTM model for sequential sleep pattern analysis
- [ ] Deploy on Streamlit Cloud
- [ ] Add weekly sleep trend tracking
- [ ] Personalized model fine-tuning per user

---

## 👤 Author

**Pabi** — [@pabi1234810](https://github.com/pabi1234810)