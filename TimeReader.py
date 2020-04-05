import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import datetime


class TimeReader:
    def __init__(self):
        self.df = pd.read_csv("TimeSeries.csv")
        self.cleaningData()

    def cleaningData(self):
        """
        Cleaning up the data
        :return:
        """
        # Fill Nulls with zeros
        self.df['hospitalizedCumulative'] = self.df['hospitalizedCumulative'].fillna(0)
        self.df['death'] = self.df['death'].fillna(0)

        # Format date and calculate days since day-0
        self.df['dateChecked'] = pd.to_datetime(self.df['dateChecked']).dt.date
        self.df['day'] = (self.df['dateChecked'] - datetime.date(2020, 4, 1)).dt.days
        print()

    def getStateData(self, state):
        """
        Get cases and deaths of state
        :param df: Dataframe main
        :param state: State input
        :return: Cases and deaths as dataframes
        """
        df = self.df
        stateData = ps.sqldf("SELECT * from df WHERE state = '" + state + "'")
        death = stateData['death']
        print(death)


if __name__ == '__main__':
    timereader = TimeReader()
    timereader.getStateData('CA')
    print()