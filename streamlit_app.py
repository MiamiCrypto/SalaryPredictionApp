import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(page_title="Salary Prediction Dashboard", page_icon="💼", layout="centered", initial_sidebar_state="expanded")

# # Custom CSS for dark mode
# dark_mode_css = """
#     <style>
#     body, .stApp {
#         background-color: #0e1117;
#         color: #ffffff;
#     }
#     .css-1d391kg, .css-1nv0d2y, .css-1v3fvcr, .css-2trqyj, .css-12oz5g7, .css-1lypyze, .css-1rs6os, .css-1vbkxwb, .css-1grh8ro {
#         color: #ffffff;
#     }
#     .css-1d3w5w1, .css-1pxd2dn, .css-1a32fsj, .css-1hfz7xv, .css-1l5zk5j, .css-18ni7ap, .css-1cpxqw2 {
#         background-color: #0e1117;
#     }
#     </style>
# """
# st.markdown(dark_mode_css, unsafe_allow_html=True)

# # Custom CSS to hide the "Made with Streamlit" watermark
# hide_streamlit_style = """
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load the original dataset to get the mean and standard deviation of the salary
csv_file_path = 'Balanced_Graduated_Data.csv'

if os.path.exists(csv_file_path):
    try:
        students_data = pd.read_csv(csv_file_path)
        salary_mean = students_data['Salary'].mean()
        salary_std = students_data['Salary'].std()
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
else:
    st.error(f"CSV file not found: {csv_file_path}")

# Load the trained model
model_file_path = 'random_forest_model.pkl'
if os.path.exists(model_file_path):
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error(f"Model file not found: {model_file_path}")

# Define the features and their possible values
features = {
    'GPA': (0.0, 4.0, 3.0),
    'Skills': [
        'Coding Skills', 'Machine Learning', 'App Dev', 'Backend', 
        'Creativity', 'Presentation Skills', 'Problem Solving', 
        'Budget Management', 'Business Understanding', 'Collaboration', 
        'Data Science', 'Decision Making', 'Improvement', 
        'Data Driven Decision Making', 'Attention to Detail', 
        'Programming Languages'
    ],
    'Major': ['Applied Artificial Intelligence', 'Data Analytics'],
    'Graduated': ['Yes', 'No']
}

# Mapping for major codes to names
major_mapping = {0: 'Applied Artificial Intelligence', 1: 'Data Analytics'}

# Streamlit UI
st.title("Salary Prediction Dashboard")
st.header("Enter the values for the following features to predict the salary")

# Display the image smaller and centered
st.image("salaryprediction.png", width=300, caption="Predict your future Salary")

# Input fields for the features
input_data = {}

# GPA as a slider
input_data['GPA'] = st.slider('GPA', *features['GPA'])

# Skills as a multi-select dropdown
selected_skills = st.multiselect('Select Skills', features['Skills'])

# Major as a dropdown
input_data['Major'] = st.selectbox('Select Major', features['Major'])

# Graduated as a dropdown
input_data['Graduated'] = st.selectbox('Graduated (Yes/No)', features['Graduated'])

# Initialize all skills to 0 (not selected)
for skill in features['Skills']:
    input_data[skill] = 0

# Set selected skills to 1 (selected)
for skill in selected_skills:
    input_data[skill] = 1

# Convert categorical features to numerical
input_data['Major'] = features['Major'].index(input_data['Major'])
input_data['Graduated'] = 1 if input_data['Graduated'] == 'Yes' else 0

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the input DataFrame has the same columns as the model expects
expected_features = model.feature_names_in_

# Reorder and align the input data to match the expected features
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Predict the salary
if st.button("Predict Salary"):
    standardized_prediction = model.predict(input_df)
    original_scale_prediction = standardized_prediction[0] * salary_std + salary_mean
    st.write(f"Predicted Salary: ${original_scale_prediction:.2f}")

# Show feature importances
if st.button("Show Feature Importances"):
    feature_importances = model.feature_importances_
    feature_names = model.feature_names_in_
    
    # Create a DataFrame for the feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Plotting the bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
    ax.set_title('Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)

# Visualizing Major Distribution and GPA Distribution side by side
st.header("Distribution of Majors and GPA")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Majors")
    students_data['Major'] = students_data['Major'].map(major_mapping)  # Map major codes to names
    major_counts = students_data['Major'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(major_counts, labels=major_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(major_counts)))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

with col2:
    st.subheader("Distribution of GPA")
    fig, ax = plt.subplots()
    sns.histplot(students_data['GPA'], bins=10, kde=True, ax=ax)
    ax.set_title('Distribution of GPA')
    ax.set_xlabel('GPA')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

st.header("Average Salary by Major")
avg_salary_by_major = students_data.groupby('Major')['Salary'].mean().reset_index()
    
# Debugging: Print the DataFrame to ensure it looks correct
st.write("Average Salary by Major DataFrame:")
#st.header("Average Salary by Major")
st.write(avg_salary_by_major)
    
if not avg_salary_by_major.empty:
    # Plotting the bar plot for average salary by major
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Salary', y='Major', data=avg_salary_by_major, palette='viridis', ax=ax)
    ax.set_title('Average Salary by Major')
    ax.set_xlabel('Average Salary')
    ax.set_ylabel('Major')
    st.pyplot(fig)
else:
    st.write("No data available to display the average salary by major.")
