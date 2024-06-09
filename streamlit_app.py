import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Load the model from the pickle file
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title('Student Salary Predictor')

# Load and display the image centered
image = Image.open('salaryprediction.png')
st.image(image, use_column_width=True, caption="Centered Image")

# GPA input
gpa = st.slider('GPA', 0.0, 4.0, 3.0)

# Skills input
skills = st.text_area('Skills (separated by commas)', 'Python, Data Analysis')

# Prepare the input data for prediction
skills_list = skills.split(',')

# Example of how you might encode skills
# This should be replaced with the actual encoding used during training
skills_encoded = [1 if skill in skills_list else 0 for skill in ['Python', 'Data Analysis', 'Machine Learning', 'Deep Learning', 'Statistics']]

input_data = pd.DataFrame([[gpa] + skills_encoded], columns=['GPA', 'Python', 'Data Analysis', 'Machine Learning', 'Deep Learning', 'Statistics'])

# Make prediction
if st.button('Predict Salary'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Salary: ${prediction[0]:,.2f}')

# Additional notes
st.write('Adjust the GPA and add relevant skills to see the predicted salary.')


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Load the trained model
# model_path = 'random_forest_model.pkl'
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# # Load the dataset for feature importance
# dataset_url = 'https://raw.githubusercontent.com/MiamiCrypto/SalaryPredictionApp/master/Students_With_Skills_Complete_Balanced.csv'
# students_data = pd.read_csv(dataset_url)
# students_data_encoded = pd.get_dummies(students_data, drop_first=True)
# exclude_columns = [col for col in students_data_encoded.columns if 'First_Name' in col or 'Email' in col]
# exclude_columns += ['Student_ID', 'Salary']
# features = students_data_encoded.drop(columns=exclude_columns).columns

# # Function to get top suggestions for improving salary
# def get_top_suggestions(features, importances, num_suggestions=5):
#     sorted_indices = np.argsort(importances)[::-1]
#     top_features = features[sorted_indices][:num_suggestions]
#     return top_features

# # Streamlit Application
# st.title("Student Salary Prediction")

# # Display an image from a file
# st.image("salaryprediction.png", width=300, caption="Predict your future Salary")

# # Input Form
# st.header("Enter Student Details for Salary Prediction")
# major = st.selectbox("Major", options=students_data['Major'].unique())
# gpa = st.slider("GPA", min_value=0.0, max_value=4.0, step=0.01)
# num_skills = st.number_input("Number of Skills", min_value=0)
# skills = st.multiselect("Skills", options=students_data.columns[10:])  # Assuming skills start from the 11th column

# # Add a dropdown for Graduated
# graduated = st.selectbox("Graduated", options=['Yes', 'No'])

# # Prepare the input data
# input_data = {feature: 0 for feature in features}
# input_data[f"Major_{major}"] = 1
# input_data["GPA"] = gpa
# input_data["Number_of_Skills"] = num_skills
# input_data[f"Graduated_{graduated}"] = 1 if graduated == 'Yes' else 0  # Correctly handle Graduated
# for skill in skills:
#     skill_col = f'Skills_{skill}' if f'Skills_{skill}' in input_data else skill
#     if skill_col in input_data:
#         input_data[skill_col] = 1
# input_df = pd.DataFrame([input_data])

# # Ensure input_df has all the necessary columns
# missing_cols = set(features) - set(input_df.columns)
# for col in missing_cols:
#     input_df[col] = 0
# input_df = input_df[features]

# # Predict Salary
# if st.button("Predict Salary"):
#     predicted_salary = model.predict(input_df)[0]
#     st.write(f"Predicted Salary: ${predicted_salary:.2f}")

#     # Provide Suggestions
#     st.header("Suggestions to Improve Salary")
#     feature_importances = model.feature_importances_
#     suggestions = get_top_suggestions(features, feature_importances)
#     st.write(f"Consider improving the following features/skills to potentially increase your salary:")
#     for suggestion in suggestions:
#         st.write(f"- {suggestion}")

# # Batch Prediction from CSV
# st.header("Batch Prediction from CSV")
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
# if uploaded_file is not None:
#     batch_data = pd.read_csv(uploaded_file)
#     batch_data_encoded = pd.get_dummies(batch_data, drop_first=True)
#     # Ensure batch_data_encoded has all the necessary columns
#     missing_cols_batch = set(features) - set(batch_data_encoded.columns)
#     for col in missing_cols_batch:
#         batch_data_encoded[col] = 0
#     batch_data_encoded = batch_data_encoded[features]
    
#     predictions = model.predict(batch_data_encoded)
#     batch_data['Predicted_Salary'] = predictions
#     st.write("Batch Predictions:")
#     st.write(batch_data)
#     st.download_button("Download Predictions", data=batch_data.to_csv(index=False), file_name="batch_predictions.csv")

# # Feature Importance Visualization
# st.header("Feature Importance")
# feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
# st.bar_chart(feature_importance_df.set_index('Feature')['Importance'])

# # Model Performance Metrics
# st.header("Model Performance Metrics")
# mae_tuned = 2169.70
# mse_tuned = 21447578.28
# r2_tuned = 0.975
# st.write(f"Mean Absolute Error (MAE): {mae_tuned:.2f}")
# st.write(f"Mean Squared Error (MSE): {mse_tuned:.2f}")
# st.write(f"R² Score: {r2_tuned:.2f}")
