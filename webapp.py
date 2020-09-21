#This program detects if someone has diabetes

#Import the libraries

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create title and subtitle

st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python.
""")

#open and display an image
# image = Image.open("insert path of image here")
# st.image(image, caption='ML', use_column_width=True)

#Get the data
df = pd.read_csv('/Users/hiyashah/PycharmProjects/diabetesapp/diabetes.csv')

#Set a subheader
st.subheader('Data Information: ')

#show data as table
st.dataframe(df)

#show statistics on the data
st.write(df.describe())

#show data as a chart
chart= st.bar_chart(df)

#split data into indep x and dependent y values

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#splitting the data set into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#getting feature input from user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness= st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin= st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    bmi= st.sidebar.slider('bmi', 0.0, 67.1, 32.0)
    diabetes_pedigree_function= st.sidebar.slider('diabetes_pedigree_function', 0.078, 2.42, 0.3725)
    age= st.sidebar.slider('age', 21, 89, 29)

    #store dictionary of values into variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'bmi': bmi,
                 'diabetes_pedigree_function': diabetes_pedigree_function,
                 'age': age
                 }

    #transform data into dataframe
    features = pd.DataFrame(user_data, index= [0])
    return features

#store user input into variable
user_input= get_user_input()

#set a subheader and display user's input
st.subheader('User Input:')
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#show model metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

#store model predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)