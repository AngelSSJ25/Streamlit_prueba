import streamlit as st
import pandas as pd
import numpy as np
import pickle  # to load a saved model
import base64  # to handle gif encoding


@st.cache_data()
def get_fvalue(val): 
    feature_dict = {"No": 1, "Yes": 2}     
    for key,value in feature_dict.items():
        if val == key:
            return value


def get_value(val, my_dict):    
    for key,value in my_dict.items():   
        if val==key:
           return my_dict[val]



app_mode = st.sidebar.selectbox("select page",["Home","Prediction"])
if app_mode=="Home":
    st.title("Loan prediction :")
    st.write("App realised by: Me")
    st.image("fp.jpeg")
    st.markdown("Dataset : ")
    data = pd.read_csv("loan.csv")
    st.write(data.head())
    st.markdown("Applicant Income Vs Loan Amount ")
    st.bar_chart(data[["ApplicantIncome", "LoanAmount"]].head(20))

if app_mode == "Prediction":   
    # Inputs numéricos
    ApplicantIncome = st.sidebar.slider('Applicant Income', 0, 10000, 0)    
    CoapplicantIncome = st.sidebar.slider('Coapplicant Income', 0, 10000, 0)
    LoanAmount = st.sidebar.slider('Loan Amount in K$', 9.0, 700.0, 200.0)    
    Loan_Amount_Term = st.sidebar.selectbox('Loan Amount Term (months)', [36, 60, 84, 120, 180, 240, 300, 360])
    Credit_History = st.sidebar.selectbox('Credit History', [0, 1])    

    # Inputs categóricos
    Gender = st.sidebar.radio('Pick your gender', ['Male', 'Female'])
    Married = st.sidebar.radio('Married?', ['Yes', 'No'])
    Education = st.sidebar.radio('Education', ['Graduate', 'Not Graduate'])
    Self_Employed = st.sidebar.radio('Self Employed?', ['Yes', 'No'])
    Dependents = st.sidebar.radio('Dependents', ['0', '1', '2', '3+'])
    Property_Area = st.sidebar.radio('Property Area', ['Rural', 'Urban', 'Semiurban'])

    # Codificación idéntica al entrenamiento
    gender_dict = {"Male":1, "Female":2}
    edu_dict = {"Graduate":1, "Not Graduate":0}
    dependents_dict = {"0":[1,0,0,0], "1":[0,1,0,0], "2":[0,0,1,0], "3+":[0,0,0,1]}
    prop_dict = {"Rural":[1,0,0], "Urban":[0,1,0], "Semiurban":[0,0,1]}

    class_0, class_1, class_2, class_3 = dependents_dict[Dependents]
    Rural, Urban, Semiurban = prop_dict[Property_Area]

    # Construir DataFrame con columnas en el mismo orden
    input_data = pd.DataFrame([[
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        gender_dict[Gender],
        1 if Married=="No" else 2,
        edu_dict[Education],
        1 if Self_Employed=="No" else 2,
        class_0, class_1, class_2, class_3,
        Rural, Urban, Semiurban
    ]], columns=[
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Dependents_0",
        "Dependents_1",
        "Dependents_2",
        "Dependents_3+",
        "Property_Area_Rural",
        "Property_Area_Urban",
        "Property_Area_Semiurban"
    ])
if st.button("Predict"):
    loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))
    proba = loaded_model.predict_proba(input_data)

    st.write("Probability of approval:", proba[0][1])

    if proba[0][1] >= 0.4:   # umbral ajustado
        st.success('Congratulations!! you will get the loan from Bank')
        st.image("https://gifdb.com/images/high/si-498-x-278-gif-g5oee22yetynrlqt.gif")
    else:
        st.error('According to our calculations, you will not get the loan from Bank')
        st.image("https://i.pinimg.com/originals/f0/b1/f2/f0b1f28bffd5b9c2fc49fd3ffc35211b.gif")
