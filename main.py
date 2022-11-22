import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import base64
import json
import requests
import time
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

col1,col2,col3 = st.columns([2,3,3])
hide_menu_style = """
        <style>
        #MainMenu,header, footer {visibility: hidden;}

        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

#st.title("WINE QUALITY PREDICTION")
col3.markdown("")
col3.markdown("")
col3.markdown("")
col3.markdown("\nFill the information given below to predict your wine quality.")

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.master('local[*]').appName("winequalitypredict").getOrCreate()

# @st.cache(suppress_st_warning=True)
def do_prediction(volatile_acidity,citric_acid,chlorides,sulphates,alcohol):


    df_pyspark1 = spark.read.csv("winequality-red.csv",header=True,inferSchema=True)
    df_pyspark = df_pyspark1.drop('fixed acidity','residual sugar','free sulfur dioxide','total sulfur dioxide','density','pH')
    X = df_pyspark.drop('quality')

    assembler = VectorAssembler(inputCols=X.columns,outputCol='features')
    output = assembler.transform(df_pyspark).select('features','quality')
    regression = LinearRegression(featuresCol='features',labelCol='quality')


    regression_model = regression.fit(output)
    # st.write("Coeff : {}".format(regression_model.coefficients))
    # st.write("Intercept : {}".format(regression_model.intercept))

    test_data = [(volatile_acidity,citric_acid,chlorides,sulphates,alcohol,)]
    rdd = spark.sparkContext.parallelize(test_data)
    col_name = ["volatile acidity","citric acid","chlorides","sulphates","alcohol"]
    test_df = spark.createDataFrame(rdd).toDF(*col_name)

    # test_assembler = VectorAssembler(inputCols=test_df.columns, outputCol='features')
    test_output = assembler.transform(test_df).select('features')
    # test_output.show()
    test_op = regression_model.transform(test_output)
    print(test_op)
    # st.write(str(test_op.show()))
        #Fetch function for animation
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    #Local function for animation
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    #files
    good =load_lottiefile("animations/good.json")
    bad =load_lottiefile("animations/bad.json")
    #links
    #good = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_2xyfgjt4.json")
    #bad = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_vncvnui2.json")
    # if(test_op.collect()[0][1]>10):
    newtest = test_op.collect()[0][1]
    col3.write(f"Prediction of wine quality is {int(newtest)}")
    with col3:
        if (int(newtest)>=5):
            st_lottie(good)
        else:
            st_lottie(bad)

with col3:    
    form = st.form(key = "my_form")

volatile_acidity = form.number_input(label="Enter Volatile acidity : ",format="%.2f",step=1.00,min_value=0.00, max_value=2.00)
citric_acid = form.number_input(label="Enter citric acid : ",format="%.2f",step=1.00,min_value=0.00, max_value=4.00)
chlorides = form.number_input(label="Enter chlorides : ",format="%.3f",step=1.000,min_value=0.000, max_value=1.000)
sulphates = form.number_input(label="Enter sulphates : ",format="%.2f",step=1.00,min_value=0.00, max_value=4.00)
alcohol = form.number_input(label="Enter alcohol : ",format="%.1f",step=1.0,min_value=1.0, max_value = 15.0)

submit = form.form_submit_button(label="Predict")

if submit:
    # st.write("Head Size is {}".format(headSize))
    do_prediction(volatile_acidity,citric_acid,chlorides,sulphates,alcohol)



##BLACKMAGIC

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('wine.png')
#Fetch function for animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
#Local function for animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
#good = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_2xyfgjt4.json")
#bad = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_vncvnui2.json")
#rocket = load_lottiefile("animations/rocket.json")
#st_lottie(good)
