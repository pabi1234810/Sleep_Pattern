import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(
    page_title="Sleep Quality Predictor",
    page_icon="🌙",
    layout="wide"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = joblib.load('models/sleep_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

LABELS = ['Poor', 'Fair', 'Good']
EMOJI  = ['🔴', '🟡', '✅']
COLORS = ['#e74c3c', '#f39c12', '#2ecc71']

# Average values used when user has no smartwatch
AVERAGE_REM   = 22
AVERAGE_DEEP  = 18
AVERAGE_LIGHT = 60

# --- Header ---
st.title("🌙 Sleep Quality Predictor")
st.markdown("Answer a few questions about your lifestyle and sleep to get your sleep quality prediction.")
st.markdown("---")

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Info")

    age = st.slider("Age", 10, 80, 25,
        help="Sleep patterns naturally change with age.")

    smoking = st.selectbox("Smoking", ["No", "Yes"],
        help="Smoking is a stimulant and disrupts deep sleep.")

    exercise = st.slider("Exercise (days/week)", 0, 7, 3,
        help="Regular exercise improves sleep quality. Aim for 3-5 days/week.")

    caffeine = st.slider("Caffeine (mg/day)", 0, 400, 50,
        help="1 coffee ≈ 95mg. Avoid caffeine after 2pm.")

    with st.expander("☕ Common caffeine amounts"):
        st.markdown("""
        - Espresso: ~63mg
        - Coffee (1 cup): ~95mg
        - Green tea: ~29mg
        - Energy drink: ~80mg
        - Cola (1 can): ~34mg
        """)

    alcohol = st.slider("Alcohol (units/week)", 0, 10, 1,
        help="Alcohol reduces REM and deep sleep quality.")

with col2:
    st.subheader("😴 Sleep Details")

    sleep_duration = st.slider("Sleep Duration (hrs)", 1.0, 12.0, 7.5,
        help="Total hours slept. Adults need 7-9 hours.")

    awakenings = st.slider("Awakenings per night", 0, 10, 2,
        help="How many times did you fully wake up? 0-2 is normal.")

    st.markdown("---")

    # --- Smartwatch Toggle ---
    has_smartwatch = st.toggle(
        "⌚ I have a smartwatch (Fitbit / Apple Watch / Garmin)",
        value=False
    )

    if has_smartwatch:
        st.success("Great! Enter your sleep stage data from your app.")

        with st.expander("ℹ️ Where to find this in your app"):
            st.markdown("""
            - **Fitbit**: Today tab → Sleep tile → tap night → scroll to Sleep Stages
            - **Apple Watch**: Health app → Browse → Sleep → Sleep Stages
            - **Garmin**: Garmin Connect app → Sleep → Sleep Stages chart
            """)

        rem_pct   = st.slider("REM Sleep % 💭", 0, 40, 22,
            help="When you dream. Important for memory. Aim for 20-25%.")
        deep_pct  = st.slider("Deep Sleep % 🌊", 0, 40, 18,
            help="Most restorative stage. Body heals. Aim for 15-20%.")
        light_pct = st.slider("Light Sleep % 😴", 0, 80, 60,
            help="Transitional stage. Usually the largest portion. Aim for 50-60%.")

    else:
        st.info("""
        **No smartwatch?** No problem!
        We'll use healthy average values for sleep stages:
        - 💭 REM: 22%
        - 🌊 Deep: 18%
        - 😴 Light: 60%
        
        For a more accurate prediction, consider using a Fitbit or Apple Watch.
        """)
        rem_pct   = AVERAGE_REM
        deep_pct  = AVERAGE_DEEP
        light_pct = AVERAGE_LIGHT

st.markdown("---")

# --- Predict Button ---
if st.button("🔍 Predict My Sleep Quality", use_container_width=True):

    smoking_encoded = 1 if smoking == "Yes" else 0
    features = np.array([[age, sleep_duration, rem_pct, deep_pct,
                          light_pct, awakenings, caffeine,
                          alcohol, exercise, smoking_encoded]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    # --- Result ---
    st.markdown("## 📋 Your Results")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Sleep Quality", f"{EMOJI[prediction]} {LABELS[prediction]}")
    with res_col2:
        st.metric("Sleep Duration", f"{sleep_duration} hrs")
    with res_col3:
        st.metric("Awakenings", awakenings)

    if not has_smartwatch:
        st.caption("⚠️ Prediction uses average sleep stage values. Connect a smartwatch for higher accuracy.")

    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    # --- Confidence Chart ---
    with chart_col1:
        st.markdown("### 📊 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(LABELS, proba, color=COLORS)
        ax.set_xlim(0, 1)
        for bar, prob in zip(bars, proba):
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{prob*100:.1f}%', va='center', fontsize=11)
        ax.set_xlabel("Confidence")
        ax.set_title("How confident is the model?")
        plt.tight_layout()
        st.pyplot(fig)

    # --- Sleep Stage Pie Chart ---
    with chart_col2:
        st.markdown("### 🧠 Sleep Stage Breakdown")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        stages = ['REM 💭', 'Deep 🌊', 'Light 😴']
        values = [rem_pct, deep_pct, light_pct]
        colors = ['#9b59b6', '#2ecc71', '#3498db']
        wedges, texts, autotexts = ax2.pie(
            values, labels=stages, colors=colors,
            autopct='%1.1f%%', startangle=90)
        if not has_smartwatch:
            ax2.set_title("Average Sleep Stage Distribution")
        else:
            ax2.set_title("Your Sleep Stage Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")

    # --- Your Sleep vs Ideal ---
    st.markdown("### 📈 Your Sleep vs Ideal")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    categories  = ['REM %', 'Deep %', 'Light %', 'Duration (hrs)', 'Awakenings']
    your_values  = [rem_pct, deep_pct, light_pct, sleep_duration * 10, awakenings * 5]
    ideal_values = [22, 18, 60, 75, 10]
    x = range(len(categories))
    ax3.bar([i - 0.2 for i in x], your_values,  width=0.4, label='You',   color='#3498db')
    ax3.bar([i + 0.2 for i in x], ideal_values, width=0.4, label='Ideal', color='#2ecc71', alpha=0.7)
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.set_title("Your Sleep Metrics vs Ideal")
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")

    # --- Personalized Suggestions ---
    st.markdown("### 💡 Personalized Suggestions")
    tips = []
    if awakenings > 3:
        tips.append(("⚠️", "Frequent awakenings",
                     "Check noise, light, or stress levels. Try white noise or blackout curtains."))
    if caffeine > 100:
        tips.append(("⚠️", "High caffeine intake",
                     "Avoid caffeine after 2pm. Caffeine has a half-life of ~6 hours."))
    if alcohol > 2:
        tips.append(("⚠️", "Alcohol affecting sleep",
                     "Alcohol reduces REM sleep. Avoid it close to bedtime."))
    if exercise == 0:
        tips.append(("⚠️", "No exercise detected",
                     "Even 20 mins of walking per day significantly improves sleep quality."))
    if has_smartwatch and deep_pct < 15:
        tips.append(("⚠️", "Low deep sleep",
                     "Try sleeping 30 mins earlier. Avoid screens 1 hour before bed."))
    if has_smartwatch and rem_pct < 20:
        tips.append(("⚠️", "Low REM sleep",
                     "Reduce alcohol and screen time before bed. Keep a consistent sleep schedule."))
    if sleep_duration < 6:
        tips.append(("⚠️", "Too little sleep",
                     "Aim for at least 7 hours. Sleep deprivation affects memory and mood."))
    if smoking == "Yes":
        tips.append(("⚠️", "Smoking affects sleep",
                     "Nicotine is a stimulant and causes lighter, more disrupted sleep."))
    if not has_smartwatch:
        tips.append(("💡", "Get a smartwatch",
                     "A Fitbit or Apple Watch can track your REM and deep sleep for more accurate predictions."))
    if not tips:
        tips.append(("✅", "Great sleep habits!", "Keep up the good work. Your sleep looks healthy!"))

    for emoji, title, detail in tips:
        st.info(f"**{emoji} {title}:** {detail}")

    # --- Sleep Score ---
    st.markdown("---")
    st.markdown("### 🏅 Your Sleep Score")
    score = int(proba[prediction] * 100)
    st.progress(score / 100)
    st.markdown(f"**{score}/100** — Based on model confidence in your predicted sleep quality.")