import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit Application
st.title("Weight Management with Burnt Calorie Predictor ML")

# Load the saved model and scaler
best_model = joblib.load("/CalorieTrack/best_model.pkl")
scaler = joblib.load("/CalorieTrack/scaler.pkl")

def predict_calories_burned(age, height, weight, exercise_level, gender, heart_rate=80):
    """Predict daily calories burned."""
    # Combine features into a single array in the order used during training
    features = np.array([[age, height, weight, exercise_level, gender, heart_rate]])
    features = scaler.transform(features)  # Scale the features
    
    return best_model.predict(features)[0]

def calculate_weeks_to_goal(current_weight, target_weight, weight_change_per_week, daily_calories_burned):
    """
    Calculate the number of weeks needed to achieve the target weight.
    Includes calorie deficit (for weight loss) or surplus (for weight gain).
    """
    total_weight_change = abs(current_weight - target_weight)  # Total weight to lose or gain
    total_calorie_change = total_weight_change * 7700  # 7700 kcal per kg of fat
    
    # Determine if it's weight loss or weight gain
    if target_weight < current_weight:  # Weight loss
        daily_calorie_deficit = weight_change_per_week * 7700 / 7  # Weekly deficit spread over 7 days
        daily_calorie_intake = daily_calories_burned - daily_calorie_deficit
        weeks = total_calorie_change / (daily_calorie_deficit * 7)
        return weeks, daily_calorie_intake, "deficit"
    else:  # Weight gain
        daily_calorie_surplus = weight_change_per_week * 7700 / 7  # Weekly surplus spread over 7 days
        daily_calorie_intake = daily_calories_burned + daily_calorie_surplus
        weeks = total_calorie_change / (daily_calorie_surplus * 7)
        return weeks, daily_calorie_intake, "surplus"

# User Inputs
st.header("Input Your Details")
gender = st.selectbox("Select your gender:", ["Male", "Female"])
age = st.number_input("Enter your age (years):", min_value=10, max_value=100, step=1)
height = st.number_input("Enter your height (cm):", min_value=100.0, max_value=250.0, step=0.1)
current_weight = st.number_input("Enter your current weight (kg):", min_value=30.0, max_value=200.0, step=0.1)
target_weight = st.number_input("Enter your target weight (kg):", min_value=30.0, max_value=200.0, step=0.1)
weight_change_per_week = st.number_input("Enter your desired weight change per week (kg):", min_value=0.1, max_value=5.0, step=0.1)
exercise_level = st.selectbox("Select your exercise level:", ["Light", "Moderate", "Intense"])

# Map gender and exercise level to numerical values
gender_numeric = 0 if gender == "Male" else 1
exercise_map = {"Light": 1, "Moderate": 2, "Intense": 3}
exercise_numeric = exercise_map[exercise_level]

# Predict and Calculate
if st.button("Calculate"):
    # Predict daily calories burned
    daily_calories_burned = predict_calories_burned(age, height, current_weight, exercise_numeric, gender_numeric)
    
    # Calculate weeks needed, daily calorie intake, and type of change (deficit/surplus)
    weeks_needed, daily_calorie_intake, change_type = calculate_weeks_to_goal(
        current_weight, target_weight, weight_change_per_week, daily_calories_burned
    )

    # Generate weekly weight data for plotting
    weeks = np.arange(0, weeks_needed + 1)  # Week numbers
    if change_type == "deficit":
        weight_change = current_weight - (weeks * weight_change_per_week)
    else:  # Surplus
        weight_change = current_weight + (weeks * weight_change_per_week)
    weight_change = np.clip(weight_change, target_weight, current_weight) if target_weight < current_weight else np.clip(weight_change, current_weight, target_weight)
    
    # Display results
    st.subheader(f"To reach your goal weight of {target_weight} kg:")
    st.write(f"- **Daily calories burned**: {daily_calories_burned:.2f} kcal")
    if change_type == "deficit":
        st.write(f"- **Suggested daily calorie intake (deficit)**: {daily_calorie_intake:.2f} kcal")
    else:
        st.write(f"- **Suggested daily calorie intake (surplus)**: {daily_calorie_intake:.2f} kcal")
    st.write(f"- **Estimated weeks to achieve the goal**: {weeks_needed:.1f}")
    
    # Plot weight change graph
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(weeks, weight_change, marker="o", linestyle="-", color="blue")
    ax.set_title("Weight Change Over Time")
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Weight (kg)")
    ax.grid(True)
    st.pyplot(fig)
