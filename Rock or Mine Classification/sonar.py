# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:17:20 2024

@author: rakei
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
import logging
log_fmt='%(asctime)s-%(name)s-%(levelname)s-%(message)s'
logging.basicConfig(level=logging.INFO,format=log_fmt,handlers=[logging.FileHandler(f"C:/Users/rakei/Downloads/logs/sonar_project.log"),logging.StreamHandler()])
logger=logging.getLogger(__name__)


spark=SparkSession.builder.master("local").appName("project").getOrCreate()

logger.info(f"Reading sonar data from the csv file")

df=pd.read_csv("Copy of sonar data.csv",header=None)

logger.info("checking the data and conducting the EDA")

logger.info(f" There are {df.shape[0]} columns and {df.shape[1]} rows")
df[60].value_counts()

df.groupby(60).mean()
logger.info("Seperating the dependent and independent columns to train the model")
df_dep=df[60]
df_ind=df.drop(columns=60,axis=1)

df_ind_train,df_ind_test,df_dep_train,df_dep_test=train_test_split(df_ind,df_dep,test_size=0.1,random_state=42)

model=LogisticRegression()
logger.info(f"We are using logistic regression for building the model")
model.fit(df_ind_train,df_dep_train)

x=model.predict(df_ind_train)
accuracy_cal=accuracy_score(x,df_dep_train)
logger.info(f"The accuracy score on training data is {accuracy_cal}")

logger.info(f"Checking the model with test data and checking the accuracy for the same")
prediction_test=model.predict(df_ind_test)
test_accuracy=accuracy_score(prediction_test,df_dep_test)

logger.info(f"The accuracy score of testing is {test_accuracy}")

logger.info(f"Building an predictive System")

input_data=(0.0303,0.0353,0.0490,0.0608,0.0167,0.1354,0.1465,0.1123,0.1945,0.2354,0.2898,0.2812,0.1578,0.0273,0.0673,0.1444,0.2070,0.2645,0.2828,0.4293,0.5685,0.6990,0.7246,0.7622,0.9242,1.0000,0.9979,0.8297,0.7032,0.7141,0.6893,0.4961,0.2584,0.0969,0.0776,0.0364,0.1572,0.1823,0.1349,0.0849,0.0492,0.1367,0.1552,0.1548,0.1319,0.0985,0.1258,0.0954,0.0489,0.0241,0.0042,0.0086,0.0046,0.0126,0.0036,0.0035,0.0034,0.0079,0.0036,0.0048)
input_data_array=np.asarray(input_data)
input_data_array_reshaped=input_data_array.reshape(1,-1)
predtiction=model.predict(input_data_array_reshaped)
print(predtiction)

if(predtiction[0]=='R'):
    print("Its a Rock")
else:
    print("Its a Mine")



