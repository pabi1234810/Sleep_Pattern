import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
    plt.title('Sleep Quality Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig('models/confusion_matrix.png')
    print("✅ Confusion matrix saved to models/")

if __name__ == "__main__":
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    joblib.dump(model, 'models/sleep_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Model saved!")

    plot_confusion_matrix(y_test, y_pred)