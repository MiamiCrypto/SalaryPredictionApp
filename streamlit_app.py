import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Load the model from the pickle file
model_path = 'random_forest_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Features used during model training, removing 'Student_ID'
all_features = [
    'Major', 'Number_of_Skills', 'Salary_Range', 'GPA',
    'Graduated', 'Algorithm Development', 'Creativity',
    'AI Ethics and Governance', 'API Development', 'Curiosity',
    'Project Management', 'Business Acumen', 'Communication',
    'AI Frameworks and Libraries', 'IoT Integration',
    'Cloud Computing', 'Programming Languages', 'Big Data Technologies',
    'Ethical Considerations', 'Data Analysis', 'Computer Vision',
    'Software Engineering Principles', 'Market Analysis', 'Adaptability',
    'Data Visualization', 'Edge Computing', 'Natural Language Processing (NLP)',
    'Problem-Solving', 'Robotics', 'Database Management', 'Data Engineering',
    'Model Training and Evaluation', 'ETL Processes', 'Deep Learning', 'DevOps',
    'Presentation Skills', 'Collaboration', 'Machine Learning Algorithms', 'Leadership',
    'Critical Thinking', 'Data Preprocessing', 'Data-Driven Decision Making',
    'Attention to Detail', 'Programming', 'Salary_Binned_30k-60k',
    'Salary_Binned_60k-90k', 'Salary_Binned_90k-120k', 'GPA_Range_0', 'GPA_Range_1',
    'GPA_Range_2', 'GPA_Range_3', 'Title_0', 'Title_1', 'Title_2', 'Title_3',
    'Title_4', 'Title_5', 'Title_6', 'Title_7', 'Title_8', 'Title_9'
]

# Title of the app
st.title('Student Salary Predictor')

# Display the image centered
st.image("salaryprediction.png", width=300, caption="Predict your future Salary")

# GPA input
gpa = st.slider('GPA', 0.0, 4.0, 3.0)

# Major input as a dropdown
majors = ['Computer Science', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering', 'Mathematics', 'Physics', 'Chemistry', 'Biology']
major = st.selectbox('Major', majors)

# Other input fields
number_of_skills = st.number_input('Number of Skills', min_value=0, max_value=50, value=5)
salary_range = st.selectbox('Salary Range', ['30k-60k', '60k-90k', '90k-120k'])
graduated = st.selectbox('Graduated', [True, False])

# Skills input with a dropdown
skills_selected = st.multiselect('Select Skills', all_features[5:45])

# Prepare the input data for prediction
skills_encoded = [1 if skill in skills_selected else 0 for skill in all_features[5:45]]

# Encode categorical variables (Salary Range and Graduated)
salary_binned = [1 if salary_range == '30k-60k' else 0, 1 if salary_range == '60k-90k' else 0, 1 if salary_range == '90k-120k' else 0]
graduated_encoded = [1 if graduated else 0]

# Example encoding for GPA Range (assuming it's based on intervals)
gpa_ranges = [0, 0, 0, 0]  # Example placeholder, should be based on actual logic
if gpa < 1.0:
    gpa_ranges[0] = 1
elif gpa < 2.0:
    gpa_ranges[1] = 1
elif gpa < 3.0:
    gpa_ranges[2] = 1
else:
    gpa_ranges[3] = 1

# Example encoding for Titles (assuming 10 possible titles)
titles_encoded = [0]*10  # Example placeholder, should be based on actual logic
# Add logic to set the appropriate title index to 1

# Assemble all input data
input_data_values = [major, number_of_skills, salary_range, gpa, graduated] + skills_encoded + salary_binned + gpa_ranges + titles_encoded

# Check the length of input data values and feature columns
st.write("Length of input data values:", len(input_data_values))
st.write("Length of feature columns:", len(all_features))

if len(input_data_values) == len(all_features):
    input_data = pd.DataFrame([input_data_values], columns=all_features)

    # Debug: Print the columns of the input data
    st.write("Input Data Columns:", input_data.columns.tolist())

    # Make prediction
    if st.button('Predict Salary'):
        try:
            prediction = model.predict(input_data)
            st.write(f'Predicted Salary: ${prediction[0]:,.2f}')
        except ValueError as e:
            st.error(f"ValueError: {e}")
else:
    st.error("The lengths of input data values and feature columns do not match.")

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
