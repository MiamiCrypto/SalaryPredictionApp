import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Set the page configuration
st.set_page_config(page_title="Fortune Teller", page_icon="💼", layout="centered", initial_sidebar_state="collapsed")

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

# Custom CSS to hide the "Made with Streamlit" watermark
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
st.markdown("<h1 style='text-align: center; color: ;'>Fortune Teller</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-weight: normal;'>Crystal Clear Salary Forecast</h3>", unsafe_allow_html=True)
#st.title("Salary Prediction Dashboard")
#st.header("<h1 style='text-align: center; color: ; '>Enter the values for the following features to predict the salary</h1>", unsafe_allow_html=True)

#col1, col2, col3= st.columns(3)
# Display the image smaller and centered

#with col2:

# Display the image smaller and centered
st.image("salarycrystalball.png", width=700, caption="")

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

    # Sort the DataFrame by importance and select the top 5 features
    top_importance_df = importance_df.sort_values(by='Importance', ascending=False).head(5)

    # Plotting the bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_importance_df, palette='viridis', ax=ax)
    ax.set_title('Top 5 Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)

# Model performance section
# st.header("Model Performance")
# # Assuming you have a test set, you can display the performance metrics
# # Here, we use some dummy values for the metrics
# # Replace these dummy values with actual test data and labels
# accuracy = 0.85
# precision = 0.80
# recall = 0.78
# f1 = 0.79
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# Add sections for input data, model parameters, model performance, and prediction results

# Input data section
st.header("Sample Data")
st.write(students_data.head())

# Visualizing Major Distribution and GPA Distribution side by side
#st.header("Distribution of Majors and GPA")

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
    ax.set_title('')
    ax.set_xlabel('GPA')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

#col1, col2 = st.columns(2)

#with col1:
st.subheader("Salary Distribution by Major")

# Description text
description_text = """
<div style='text-align: center; font-size:12px;'>
Applied Artificial Intelligence shows a higher median salary compared to Data Analytics.
<div>
"""
st.markdown(description_text, unsafe_allow_html=True)

# Display the image using its actual size
st.image("Salary Distribution by Major.png", caption="")

#with col2:
st.subheader("GPA vs Salary")

# Description text
description_text = """
<div style='text-align: center; font-size:12px;'>
Indicating that higher GPAs generally correlate with higher salaries.
<div>
"""
st.markdown(description_text, unsafe_allow_html=True)

# Display the image using its actual size
st.image("GPA vs Salary.png", width = 650, caption="")

#col1, col2 = st.columns(2)
#with col1:
    
# Correlation heatmap for top features
st.subheader("Correlation Heatmap for Top Features")
corr_matrix = students_data.corr()
top_features = corr_matrix['Salary'].abs().sort_values(ascending=False).head(10).index
top_corr = students_data[top_features].corr()
    
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(top_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title('High correlations between certain skills and salary suggest their significant impact on salary prediction.')
st.pyplot(fig)

# Display the image using its actual size
st.image("Correlation Heatmap.png", caption="")

st.subheader("Skills Frequency Word Cloud")
# Description text
description_text = """
<div style='text-align: center; font-size:12px;'>
Highlights the most prevalent skills in the dataset, such as Collaboration and Presentation Skills.
<div>
"""
st.markdown(description_text, unsafe_allow_html=True)

# Display the image using its actual size
st.image("Skills Frequency Word Cloud.png", caption="")

st.subheader("Skills Distribution by Major")
# Display the image using its actual size
st.image("Skills Distribution by Major.png", caption="")


# Number of skills by major
st.subheader("Number of Skills by Major")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(y='Major', data=students_data, palette='viridis', ax=ax)
ax.set_title('It indicates that Applied Artificial Intelligence students tend to have a higher number of skills compared to Data Analytics students.')
ax.set_xlabel('Count')
ax.set_ylabel('Major')
st.pyplot(fig)

