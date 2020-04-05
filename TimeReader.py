import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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
        day0 = datetime.date(2020, 1, 15)
        self.df['dateChecked'] = pd.to_datetime(self.df['dateChecked']).dt.date
        self.df['day'] = (self.df['dateChecked'] - day0).dt.days
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
        return stateData


class SigmoidFitter:
    """
    https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
    Initial guess reference
    https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d
    """
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def fit(self):
        p0 = [2, 100, 20000]  # this is an mandatory initial guess
        popt, pcov = curve_fit(sigmoid, self.xdata, self.ydata, p0=p0)
        y = sigmoid(self.xdata, *popt)
        return y, popt


def sigmoid(x, a, x0, L):
    return L / (1 + np.exp(-(x - x0) / a))


def main():
    timereader = TimeReader()
    stateData = timereader.getStateData('CA')
    death = stateData['death']
    day = stateData['day']

    # Curve Fitting
    sigmoidfitter = SigmoidFitter(xdata=day, ydata=death)
    ypredict, popt = sigmoidfitter.fit()
    print("Total deaths: " + str(popt[2]))
    print("Half point: " + str(popt[1]))

    # Plot
    plt.scatter(day, death)
    x = np.linspace(0, 200, 10000)
    y = sigmoid(x, *popt)
    plt.plot(x, y)
    plt.xlim(45, 100)
    plt.ylim(0, 500)
    plt.show()
    exit()


if __name__ == '__main__':
    main()