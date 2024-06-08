import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the trained model
model_path = 'random_forest_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the dataset for feature importance
dataset_url = 'https://raw.githubusercontent.com/MiamiCrypto/Capstone-Project-/main/Students_With_Skills_Complete.csv'
students_data = pd.read_csv(dataset_url)
students_data_encoded = pd.get_dummies(students_data, drop_first=True)
exclude_columns = [col for col in students_data_encoded.columns if 'First_Name' in col or 'Email' in col]
exclude_columns += ['Student_ID', 'Salary']
features = students_data_encoded.drop(columns=exclude_columns).columns

# Function to get top suggestions for improving salary
def get_top_suggestions(features, importances, num_suggestions=5):
    sorted_indices = np.argsort(importances)[::-1]
    top_features = features[sorted_indices][:num_suggestions]
    return top_features

# Streamlit Application
st.title("Student Salary Prediction")

# Display an image from a file
st.image("salaryprediction.png", width=300, caption="Predict your future Salary")

# Input Form
st.header("Enter Student Details for Salary Prediction")
major = st.selectbox("Major", options=students_data['Major'].unique())
gpa = st.slider("GPA", min_value=0.0, max_value=4.0, step=0.01)
num_skills = st.number_input("Number of Skills", min_value=0)
skills = st.multiselect("Skills", options=students_data.columns[10:])  # Assuming skills start from the 11th column

# Prepare the input data
input_data = {feature: 0 for feature in features}
input_data[f"Major_{major}"] = 1
input_data["GPA"] = gpa
input_data["Number_of_Skills"] = num_skills
for skill in skills:
    if skill in input_data:
        input_data[skill] = 1
input_df = pd.DataFrame([input_data])

# Predict Salary
if st.button("Predict Salary"):
    predicted_salary = model.predict(input_df)[0]
    st.write(f"Predicted Salary: ${predicted_salary:.2f}")

    # Provide Suggestions
    st.header("Suggestions to Improve Salary")
    feature_importances = model.feature_importances_
    suggestions = get_top_suggestions(features, feature_importances)
    st.write(f"Consider improving the following features/skills to potentially increase your salary:")
    for suggestion in suggestions:
        st.write(f"- {suggestion}")

# Batch Prediction from CSV
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    batch_data_encoded = pd.get_dummies(batch_data, drop_first=True)
    predictions = model.predict(batch_data_encoded)
    batch_data['Predicted_Salary'] = predictions
    st.write("Batch Predictions:")
    st.write(batch_data)
    st.download_button("Download Predictions", data=batch_data.to_csv(index=False), file_name="batch_predictions.csv")

# Feature Importance Visualization
st.header("Feature Importance")
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
st.bar_chart(feature_importance_df.set_index('Feature')['Importance'])

# Model Performance Metrics
st.header("Model Performance Metrics")
mae_tuned = 2169.70
mse_tuned = 21447578.28
r2_tuned = 0.975
st.write(f"Mean Absolute Error (MAE): {mae_tuned:.2f}")
st.write(f"Mean Squared Error (MSE): {mse_tuned:.2f}")
st.write(f"R² Score: {r2_tuned:.2f}")

# Image Upload
st.header("Upload an Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
