import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras import utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Neural:
    """
    Neural Network for county level data
    """

    def cleaning(self):
        hospital_df = pd.read_csv("Hospitals_County_COVID_4.10.20.csv")
        medicare_df = pd.read_csv("Medicare_2017.csv")
        # medicare_df = medicare_df.head(1000)
        column_list = ['Zip_Code', 'NUM_HCPCS', 'NUM_Services', 'NUM_Medicare_BEN',
                       'Total_Submitted_Charge_Amount', 'Total_Medicare_Allowed_Amount',
                       'Total_Medicare_Payment_Amount',
                       'Total_Medicare_Standardized_Payment_Amount',
                       'NUM_HCPCS_Associated_With_Drug_Services', 'NUM_Drug_Services',
                       'NUM_Medicare_BEN_With_Drug_Services',
                       'Total_Drug_Submitted_Charge_Amount',
                       'Total_Drug_Medicare_Allowed_Amount',
                       'Total_Drug_Medicare_Payment_Amount',
                       'Total_Drug_Medicare_Standardized_Payment_Amount',
                       'Medical_Suppress_Indicator',
                       'NUM_HCPCS_Associated_With_Medical_Services', 'NUM_Medical_Services',
                       'NUM_Medicare_BEN_With_Medical_Services',
                       'Total_Medical_Submitted_Charge_Amount',
                       'Total_Medical_Medicare_Allowed_Amount',
                       'Total_Medical_Medicare_Payment_Amount',
                       'Total_Medical_Medicare_Standardized_Payment_Amount',
                       'Average_Age_of_BEN', 'NUM_BEN_Age_Less_65', 'NUM_BEN_Age_65_to_74',
                       'NUM_BEN_Age_75_to_84', 'NUM_BEN_Age_Greater_84', 'NUM_Female_BEN',
                       'NUM_Male_BEN', 'NUM_Non-Hispanic_White_BEN',
                       'NUM_Black_or_African_American_BEN', 'NUM_Asian_Pacific_Islander_BEN',
                       'NUM_Hispanic_BEN', 'NUM_American_IndianAlaska_Native_BEN',
                       'NUM_BEN_With_Race_Not_Elsewhere_Classified',
                       'NUM_BEN_With_Medicare_Only_Entitlement',
                       'NUM_BEN_With_Medicare_Medicaid_Entitlement',
                       'PCT_BEN_Atrial_Fibrillation', 'PCT_BEN_Alzheimers_Disease_or_Dementia',
                       'PCT_BEN_Asthma', 'PCT_BEN_Cancer', 'PCT_BEN_Heart_Failure',
                       'PCT_BEN_Chronic_Kidney_Disease',
                       'PCT_BEN_Chronic_Obstructive_Pulmonary_Disease', 'PCT_BEN_Depression',
                       'PCT_BEN_Diabetes', 'PCT_BEN_Hyperlipidemia', 'PCT_BEN_Hypertension',
                       'PCT_BEN_Ischemic_Heart_Disease', 'PCT_BEN_Osteoporosis',
                       'PCT_BEN_Rheumatoid_Arthritis_Osteoarthritis',
                       'PCT_BEN_Schizophrenia_Other_Psychotic_Disorders', 'PCT_BEN_Stroke',
                       'Average_HCC_Risk_Score_of_BEN']
        medicare_df = medicare_df[column_list]
        medicare_df = medicare_df.fillna(0)

        # Remove invalid zip codes and convert to integer
        # medicare_df = medicare_df.drop(medicare_df[medicare_df['Zip_Code'].str.isnumeric() == False].index)
        medicare_df['Zip_Code'] = medicare_df['Zip_Code'].astype(str).str.slice(stop=5)
        medicare_df['Zip_Code'] = pd.to_numeric(medicare_df['Zip_Code'], errors='coerce', downcast='integer')

        sum = medicare_df.groupby(by='Zip_Code').sum().reset_index()
        average = medicare_df.groupby(by='Zip_Code').mean().reset_index()

        # Inner join
        merge_df = pd.merge(medicare_df, hospital_df, how='inner', left_on='Zip_Code', right_on='ZIP')
        merge_df.head()
        return merge_df

    def nnDataPrep(self, hospital_df):
        x_list = {'LATITUDE',
                  'LONGITUDE', 'POPULATION', 'BEDS', 'HELIPAD', 'POPULATION_COUNTY'}
        X = hospital_df[x_list]
        y = hospital_df['COVID CASES 4.10.20']
        return X, y

    def nn(self, X, y):
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        scaler = StandardScaler().fit(X_train)
        x_train_scaled = scaler.transform(X_train)
        x_test_scaled = scaler.transform(X_test)

        model = models.Sequential()
        model.add(layers.Dense(16, activation=tf.nn.relu, input_dim=X_train.shape[1]))
        model.add(layers.Dense(8, activation=tf.nn.relu))
        model.add(layers.Dense(1))

        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=100)

        model.evaluate(x_test_scaled, y_test)
        print()


if __name__ == '__main__':
    neural = Neural()
    neural.cleaning()
    exit()
