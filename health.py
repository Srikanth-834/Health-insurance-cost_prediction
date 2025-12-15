import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the insurance dataset
medical_df = pd.read_csv( r"C:\Users\Srikanth\Desktop\\Health-Insurance-Cost-Prediction/insurance.csv")


# Replace categorical values with numerical values for 'sex', 'smoker', and 'region'
medical_df['sex'] = medical_df['sex'].map({'male': 0, 'female': 1})
medical_df['smoker'] = medical_df['smoker'].map({'yes': 1, 'no': 0})
medical_df['region'] = medical_df['region'].map({
    'southeast': 0,
    'southwest': 1,
    'northwest': 2,
    'northeast': 3
})
# Split dataset into features (X) and target (y)
X = medical_df.drop('charges', axis=1)
y = medical_df['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Train a linear regression model
lg = LinearRegression()
lg.fit(X_train, y_train)


# Function to predict insurance charges based on user input
def predict_charges(age, sex, bmi, children, smoker, region):
    try:
        # Convert input values to a numpy array
        np_df = np.array([[age, sex, bmi, children, smoker, region]])

        # Predict insurance charges
        prediction = lg.predict(np_df)

        return prediction[0]

    except ValueError:
        return "Please enter numerical values"


# Streamlit web app
def main():
    st.title("Medical Insurance Prediction Model")

    # Input fields for each feature
    age = st.number_input("Enter age", min_value=0, max_value=100)
    sex = st.selectbox("Select sex", options=["Male", "Female"])
    bmi = st.number_input("Enter BMI (Body Mass Index)", min_value=10.0, max_value=50.0, step=0.1)
    children = st.number_input("Enter number of children", min_value=0, max_value=10)
    smoker = st.selectbox("Smoker?", options=["Yes", "No"])
    region = st.selectbox("Select region", options=["Southeast", "Southwest", "Northwest", "Northeast"])

    # Convert categorical inputs to numerical values
    sex_num = 0 if sex == "Male" else 1
    smoker_num = 0 if smoker == "Yes" else 1
    region_num = {"Southeast": 0, "Southwest": 1, "Northwest": 2, "Northeast": 3}[region]

    if st.button("Predict"):
        prediction = predict_charges(age, sex_num, bmi, children, smoker_num, region_num)
        st.write("Predicted Medical Insurance Charges:", prediction)


if __name__ == '__main__':
    main()
