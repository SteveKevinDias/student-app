import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

#connection code
uri = st.secrets["mongo"]["uri"]
client = MongoClient(uri, server_api=ServerApi('1'))

#Create db
db = client['student']
#create collection
collection = db["student-pred"]


def load_model():
    with open("linear_reg1_proj.pkl","rb") as file:
        model,scaler,le =pickle.load(file) # file is going to return model,scaler and le
    return model,scaler,le

#function to store data and prediction in mongodb


def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities']=le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

# now lets create a UI
def main():
    st.title("Student performance Prediction") # Title for the app
    st.write("Enter your data to get a prediction for your performance") # Normal text

    hours_studied = st.number_input("Hours Studied",min_value = 0,max_value = 24)
    previous_score = st.number_input("Previous Score",min_value = 0,max_value = 100)
    extra = st.selectbox("Extra curricular Activities",["Yes","No"])
    sleep = st.number_input("Sleeping Hours",min_value = 0 , max_value = 24)
    sample_paper = st.number_input("Number of question paper solved",min_value = 0,max_value = 100)

    if st.button("Predict your score"):
        user_data = {              #data mapping
            "Hours Studied":hours_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleep,
            "Sample Question Papers Practiced":sample_paper
        }

        prediction = predict_data(user_data)
        #to insert each data and predictioon into mongodb
        user_data['prediction'] = round(float(prediction[0]),2)
        user_data = {key: int(value) if isinstance(value,np.integer) else float(value) if isinstance(value,np.floating) else value for key,value in user_data.items()}
        collection.insert_one(user_data)

        st.success(f"Your prediction result is {prediction}")





if __name__ ==  "__main__":
    main()



