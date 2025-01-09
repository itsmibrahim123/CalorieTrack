import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("CalorieTrack/calories_combined.csv")

# Convert string to binary for gender
df.replace({"male": 0, "female": 1}, inplace=True)

# Exploratory Data Analysis (Optional Visualization Section)
sb.scatterplot(x="Height", y="Weight", data=df)
plt.title("Height vs Weight")
plt.show()

features = ["Age", "Height", "Weight", "Duration"]

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sb.scatterplot(x=col, y="Calories", data=df.sample(1000))
    plt.title(f"{col} vs Calories")
plt.tight_layout()
plt.show()

float_features = df.select_dtypes(include="float").columns

plt.subplots(figsize=(15, 10))
for i, col in enumerate(float_features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Preprocess the data
features = df.drop(["User_ID", "Calories"], axis=1)
target = df["Calories"].values

# Split the dataset
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train multiple models
models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]
best_model = None
best_val_error = float("inf")

for model in models:
    model.fit(X_train, Y_train)
    train_error = mae(Y_train, model.predict(X_train))
    val_error = mae(Y_val, model.predict(X_val))
    print(f"{model}:\nTraining Error: {train_error}\nValidation Error: {val_error}\n")

    if val_error < best_val_error:
        best_val_error = val_error
        best_model = model

# Save the best model and scaler
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Streamlit Application
st.title("Weight Management with Burnt Calorie Predictor ML")

# Load the saved model and scaler
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to predict daily calories burned
def predict_calories_burned(age, height, weight, exercise_level, gender, heart_rate=80):
    features = np.array([[age, height, weight, exercise_level, gender, heart_rate]])
    features = scaler.transform(features)  # Scale the features
    return best_model.predict(features)[0]

# Function to calculate weeks to reach goal
def calculate_weeks_to_goal(current_weight, target_weight, weight_change_per_week, daily_calories_burned):
    total_weight_change = abs(current_weight - target_weight)
    total_calorie_change = total_weight_change * 7700  # 7700 kcal per kg of fat

    if target_weight < current_weight:  # Weight loss
        daily_calorie_deficit = weight_change_per_week * 7700 / 7
        daily_calorie_intake = daily_calories_burned - daily_calorie_deficit
        weeks = total_calorie_change / (daily_calorie_deficit * 7)
        return weeks, daily_calorie_intake, "deficit"
    else:  # Weight gain
        daily_calorie_surplus = weight_change_per_week * 7700 / 7
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
    daily_calories_burned = predict_calories_burned(age, height, current_weight, exercise_numeric, gender_numeric)
    weeks_needed, daily_calorie_intake, change_type = calculate_weeks_to_goal(
        current_weight, target_weight, weight_change_per_week, daily_calories_burned
    )

    # Generate weekly weight data for plotting
    weeks = np.arange(0, weeks_needed + 1)  # Week numbers
    if change_type == "deficit":
        weight_change = current_weight - (weeks * weight_change_per_week)
    else:
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
