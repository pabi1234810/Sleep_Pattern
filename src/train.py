import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ['Poor', 'Fair', 'Good']

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=LABELS, yticklabels=LABELS,
                cmap='Blues')
    plt.title('Sleep Quality Confusion Matrix (XGBoost)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig('models/confusion_matrix_xgb.png')
    print("✅ Confusion matrix saved!")

def plot_feature_importance(model):
    features = [
        'age', 'sleep_duration', 'rem_pct', 'deep_pct',
        'light_pct', 'awakenings', 'caffeine',
        'alcohol', 'exercise', 'smoking'
    ]
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)),
            importance[indices],
            color='steelblue')
    plt.xticks(range(len(features)),
               [features[i] for i in indices],
               rotation=45, ha='right')
    plt.title('Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    print("✅ Feature importance chart saved!")

if __name__ == "__main__":
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tune XGBoost with GridSearch
    print("🔍 Tuning hyperparameters...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb = XGBClassifier(
        eval_metric='mlogloss',
        random_state=42
    )

    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV accuracy: {grid_search.best_score_*100:.2f}%")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n📊 Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    joblib.dump(best_model, 'models/sleep_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Model saved!")

    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(best_model)