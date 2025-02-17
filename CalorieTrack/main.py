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

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('CalorieTrack/calories_combined.csv')

#converting string to binary
df.replace({"male": 0, "female": 1}, inplace=True)
features = df.drop(["User_ID", "Calories"], axis=1)
target = df["Calories"].values

sb.scatterplot(x='Height', y='Weight', data=df)
plt.show()

features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x)
plt.tight_layout()
plt.show()


features = df.select_dtypes(include='float').columns

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# Preprocess the data
df.replace({'male': 0, 'female': 1}, inplace=True)
features = df.drop(['User_ID', 'Calories'], axis=1)
target = df['Calories'].values

# Split the dataset
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train multiple models
models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]
for i in range(len(models)):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]}:')
    print('Training Error:', mae(Y_train, models[i].predict(X_train)))
    print('Validation Error:', mae(Y_val, models[i].predict(X_val)))
    print()

# Select the best model (e.g., XGBRegressor in this case)
best_model = models[1]

# Save the model and scaler
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Streamlit Application
# Streamlit Application
st.title("Weight Loss with Burnt Calorie Predictor ML")

# Load the saved model and scaler
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_calories_burned(age, height, weight, duration, exercise_level, gender, heart_rate=80):
    """Predict daily calories burned."""
    # Combine features into a single array in the order used during training
    features = np.array([[age, height, weight, duration, exercise_level, gender, heart_rate]])
    features = scaler.transform(features)  # Scale the features
    
    return best_model.predict(features)[0]

def calculate_weeks_to_goal(current_weight, target_weight, weight_loss_per_week):
    """Calculate the number of weeks needed to achieve the target weight."""
    total_weight_loss = current_weight - target_weight
    weeks = total_weight_loss / weight_loss_per_week
    return weeks

# User Inputs
st.header("Input Your Details")
gender = st.selectbox("Select your gender:", ["Male", "Female"])
age = st.number_input("Enter your age (years):", min_value=10, max_value=100, step=1)
height = st.number_input("Enter your height (cm):", min_value=100.0, max_value=250.0, step=0.1)
current_weight = st.number_input("Enter your current weight (kg):", min_value=30.0, max_value=200.0, step=0.1)
target_weight = st.number_input("Enter your target weight (kg):", min_value=30.0, max_value=200.0, step=0.1)
weight_loss_per_week = st.number_input("Enter your desired weight loss per week (kg):", min_value=0.1, max_value=5.0, step=0.1)
duration = st.number_input("Enter the exercise duration (minutes):", min_value=10, max_value=180, step=1)
exercise_level = st.selectbox("Select your exercise level:", ["Light", "Moderate", "Intense"])

# Map gender and exercise level to numerical values
gender_numeric = 0 if gender == "Male" else 1
exercise_map = {"Light": 1, "Moderate": 2, "Intense": 3}
exercise_numeric = exercise_map[exercise_level]

# Predict and Calculate
if st.button("Calculate"):
    daily_calories_burned = predict_calories_burned(age, height, current_weight, duration, exercise_numeric, gender_numeric)
    weeks_needed = calculate_weeks_to_goal(current_weight, target_weight, weight_loss_per_week)
    total_calorie_deficit = 7700 * (current_weight - target_weight)
    daily_calorie_deficit = total_calorie_deficit / (weeks_needed * 7)
    daily_calorie_intake = daily_calories_burned - daily_calorie_deficit

    # Display results
    st.subheader(f"To reach your goal weight of {target_weight} kg:")
    st.write(f"- **Daily calories burned**: {daily_calories_burned:.2f} kcal")
    st.write(f"- **Suggested daily calorie intake**: {daily_calorie_intake:.2f} kcal")
    st.write(f"- **Estimated weeks to achieve the goal**: {weeks_needed:.1f}")
