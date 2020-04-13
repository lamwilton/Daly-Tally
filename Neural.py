import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PCA import PCA


class Neural:
    """
    Neural Network for county level data
    """

    def __init__(self):
        """
        https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/
        """
        self.population_df = pd.read_csv("covid_county_population_usafacts.csv")
        self.confirmed_df = pd.read_csv("covid_confirmed_usafacts.csv")
        self.death_df = pd.read_csv("covid_deaths_usafacts.csv")
        self.daly_df = pd.read_csv("Data.csv")

    def cleaning(self):
        # Date format only works for Windows
        today = datetime.date(2020, 4, 8)
        todaystr = today.strftime("%#m/%#d/%y")

        # Remove unallocated counties, find days since first blood
        self.confirmed_df = self.confirmed_df[self.confirmed_df['countyFIPS'] != 0]
        self.confirmed_df['daysSinceFirst'] = (today - datetime.date(2020, 1, 22)).days - \
                                              self.confirmed_df.isin([0]).sum(axis=1) + 1
        # Remove all the previous date data, calculate cases per 100000 population
        self.confirmed_df = self.confirmed_df[
            ['countyFIPS', 'County Name', 'State', 'stateFIPS', 'daysSinceFirst', todaystr]]
        self.confirmed_df['population'] = self.population_df['population']
        self.confirmed_df['per1e5'] = self.confirmed_df[todaystr] / self.population_df['population'] * 100000

        self.death_df = self.death_df[self.death_df['countyFIPS'] != 0]
        self.death_df['daysSinceFirst'] = (today - datetime.date(2020, 1, 22)).days - \
                                          self.death_df.isin([0]).sum(axis=1) + 1
        self.death_df = self.death_df[['countyFIPS', 'County Name', 'State', 'stateFIPS', 'daysSinceFirst', todaystr]]
        self.death_df['population'] = self.population_df['population']
        self.death_df['per1e5'] = self.death_df[todaystr] / self.population_df['population'] * 100000

        self.population_df = self.population_df[self.population_df['countyFIPS'] != 0]

    def daly(self):
        """
        Join county with daly data
        :return:
        """
        daly_columns_list = ['state', 'TotalPop2018', '15-49yearsAllcauses',
                             '15-49yearsAsthma', '15-49yearsChronickidneydisease',
                             '15-49yearsChronicobstructivepulmonarydisease',
                             '15-49yearsDiabetesmellitus',
                             '15-49yearsInterstitiallungdiseaseandpulmonarysarcoidosis',
                             '15-49yearsIschemicheartdisease', '15-49yearsNeoplasms',
                             '15-49yearsOtherchronicrespiratorydiseases',
                             '15-49yearsRheumaticheartdisease', '15-49yearsStroke',
                             '50-69yearsAllcauses', '50-69yearsAsthma',
                             '50-69yearsChronickidneydisease',
                             '50-69yearsChronicobstructivepulmonarydisease',
                             '50-69yearsDiabetesmellitus',
                             '50-69yearsInterstitiallungdiseaseandpulmonarysarcoidosis',
                             '50-69yearsIschemicheartdisease', '50-69yearsNeoplasms',
                             '50-69yearsOtherchronicrespiratorydiseases',
                             '50-69yearsRheumaticheartdisease', '50-69yearsStroke',
                             '70+yearsAllcauses', '70+yearsAsthma', '70+yearsChronickidneydisease',
                             '70+yearsChronicobstructivepulmonarydisease',
                             '70+yearsDiabetesmellitus',
                             '70+yearsInterstitiallungdiseaseandpulmonarysarcoidosis',
                             '70+yearsIschemicheartdisease', '70+yearsNeoplasms',
                             '70+yearsOtherchronicrespiratorydiseases',
                             '70+yearsRheumaticheartdisease', '70+yearsStroke', 'AllAgesAllcauses',
                             'AllAgesAsthma', 'AllAgesChronickidneydisease',
                             'AllAgesChronicobstructivepulmonarydisease', 'AllAgesDiabetesmellitus',
                             'AllAgesInterstitiallungdiseaseandpulmonarysarcoidosis',
                             'AllAgesIschemicheartdisease', 'AllAgesNeoplasms',
                             'AllAgesOtherchronicrespiratorydiseases',
                             'AllAgesRheumaticheartdisease', 'AllAgesStroke', 'AllAgesTotal',
                             'Airpollution', 'Highbody-massindex', 'Highfastingplasmaglucose',
                             'HighLDLcholesterol', 'Highsystolicbloodpressure',
                             'Impairedkidneyfunction', 'Noaccesstohandwashingfacility', 'Smoking',
                             'Log10Pop', 'PercentUrbanPop', 'Density(P/mi2)', 'Adults19-25',
                             'Adults26-34', 'Adults35-54', 'Adults55-64', '65+',
                             'DaysSinceInfection', 'LandArea(mi2)', 'Children0-18', 'Allriskfactors']
        join_df = pd.merge(neural.confirmed_df, self.daly_df[daly_columns_list], how='inner', left_on='State',
                           right_on='state')
        return join_df

    def nnDataPrep(self, df):
        x_list = {'Airpollution', 'Highbody-massindex', 'Highfastingplasmaglucose',
                  'HighLDLcholesterol', 'Highsystolicbloodpressure',
                  'Impairedkidneyfunction', 'Noaccesstohandwashingfacility', 'Smoking',
                  'PercentUrbanPop', 'Density(P/mi2)', 'Adults19-25', 'Adults26-34',
                  'Adults35-54', 'Adults55-64', '65+', 'DaysSinceInfection',
                  'LandArea(mi2)', 'Children0-18', 'Allriskfactors'}
        X = df[x_list]
        y = df['positive']
        return X, y

    def nn(self, X, y):
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        scaler = StandardScaler().fit(X_train)
        x_train_scaled = scaler.transform(X_train)
        x_test_scaled = scaler.transform(X_test)

        model = models.Sequential()
        model.add(layers.Dense(16, activation=tf.nn.relu, input_dim=X_train.shape[1]))
        model.add(layers.Dense(16, activation=tf.nn.relu))
        model.add(layers.Dense(1))

        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=100)

        model.evaluate(x_test_scaled, y_test)
        print()


if __name__ == '__main__':
    pca = PCA()
    pca.readData()
    df = pca.getdf()
    neural = Neural()
    X, y = neural.nnDataPrep(df)
    neural.nn(X, y)
    exit()
