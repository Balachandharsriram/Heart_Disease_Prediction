import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title and sidebar menu
st.header("Machinelearning Prediction Model")
with st.sidebar:
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    lottie_codeing = load_lottiefile("C:\streamlit\Animation - 1712666514515.json")
    st.title("        Welcome👋")
    st_lottie(lottie_codeing,speed=1,reverse=False,loop=True,quality="low",height=None,width=None,key=None,)
    choice=st.radio("Menu", ["Home", "Train Model", "Predict"])
    
# Train Model page
if choice == "Train Model":
    st.header("Train a Machine Learning Model")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        st.write("File uploaded successfully.")

        # Read the data
        df = pd.read_csv(uploaded_file)

        # Display the first few rows of the dataframe
        st.write("Sample data:")
        st.write(df.head())

        # Select target column
        target_column = st.selectbox("Select target column", df.columns)

        # Display heatmap for the correlation matrix
        st.subheader("Correlation Heatmap")
        corr_matrix = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Split data into features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest Classifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Model Accuracy:", accuracy)

        # Feature importance
        st.header("Feature Importance")
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(feature_importance)

# Predict page
if choice == "Predict":
    loaded_model = pickle.load(open('C:/streamlit/trained_model.sav','rb'))

    # creating a function for Prediction
    def Heartdisease_prediction(input_data):
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = loaded_model.predict(input_data_reshaped)

        if (prediction[0] == 0):
            return 'The person is not Having a Heart Disease'
        else:
            return 'The person is have a Heart Disease'
    
    def main():
        # giving a title
        st.title('Heart Disease Prediction Web App')
        
        # getting the input data from the user
        anaemia = st.text_input('Number of anaemia')
        creatinine_phosphokinase = st.text_input('creatinine phosphokinase')
        high_blood_pressure = st.text_input('Blood Pressure value')
        DEATH_EVENT = st.text_input('Death event')
        smoking = st.text_input('smoking')
        sex = st.text_input('Gender')
        diabetes = st.text_input('Diabetes value')
        age = st.text_input('Age of the Person')
        ejection_fraction = st.text_input('Ejection Fraction')
        platelets = st.text_input('Platelets Level')
        serum_creatinine = st.text_input('Serum Creatine Level')
        serum_sodium = st.text_input('Serum Sodium Level')
        # code for Prediction
        Heartdisease = ''
        
        # creating a button for Prediction
        if st.button('Test Result'):
            Heartdisease = Heartdisease_prediction([anaemia, platelets,serum_creatinine,serum_sodium,creatinine_phosphokinase, high_blood_pressure, DEATH_EVENT, smoking, sex, diabetes, age,ejection_fraction])
            
        st.success(Heartdisease)
    
    main()

