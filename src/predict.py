import numpy as np
import joblib

LABELS = ['Poor', 'Fair', 'Good']
EMOJI = ['🔴', '🟡', '✅']

def predict_sleep(age, sleep_duration, rem_pct, deep_pct,
                  light_pct, awakenings, caffeine,
                  alcohol, exercise, smoking):

    model = joblib.load('models/sleep_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    smoking_encoded = 1 if smoking.lower() == 'yes' else 0

    features = np.array([[age, sleep_duration, rem_pct, deep_pct,
                          light_pct, awakenings, caffeine,
                          alcohol, exercise, smoking_encoded]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    print("\n🌙 Sleep Quality Prediction")
    print("=" * 35)
    print(f"  Result:  {EMOJI[prediction]} {LABELS[prediction]}")
    print(f"\n  Confidence:")
    for i, label in enumerate(LABELS):
        bar = '█' * int(proba[i] * 20)
        print(f"  {label:<6} {bar} {proba[i]*100:.1f}%")

    print("\n💡 Suggestions:")
    if awakenings > 3:
        print("  ⚠️  Frequent awakenings — check sleep environment")
    if caffeine > 100:
        print("  ⚠️  High caffeine — avoid after 2pm")
    if alcohol > 2:
        print("  ⚠️  Alcohol disrupts REM sleep")
    if exercise == 0:
        print("  ⚠️  No exercise — even 20 mins/day improves sleep")
    if deep_pct < 15:
        print("  ⚠️  Low deep sleep — try sleeping 30 mins earlier")

if __name__ == "__main__":
    # Example prediction — change these values to test!
    predict_sleep(
        age=28,
        sleep_duration=7.5,
        rem_pct=22,
        deep_pct=18,
        light_pct=60,
        awakenings=2,
        caffeine=50,
        alcohol=1,
        exercise=3,
        smoking='No'
    )