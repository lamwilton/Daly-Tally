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
    Fitting the data to the sigmoid function and plot it
    https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
    Initial guess reference
    https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d
    """
    def __init__(self, xdata, ydata, state):
        self.xdata = xdata
        self.ydata = ydata
        self.state = state
        self.popt = np.empty_like
        self.pcov = np.empty_like

    def sigmoid(self, x, a, x0, L):
        """
        Sigmoid function
        :param x:
        :param a: some constant
        :param x0: x shift
        :param L: Max
        :return:
        """
        return L / (1 + np.exp(-(x - x0) / a))

    def fit(self):
        """
        Curve fitting
        :return:
        """
        # Error function sigma to apply greater weights to the recent points
        temp = np.arange(0, np.size(self.xdata))
        sigma = 1 / temp ** 2

        p0 = [1, 80, 2000]  # this is an mandatory initial guess
        self.popt, self.pcov = curve_fit(self.sigmoid, self.xdata, self.ydata, p0=p0, sigma=sigma)

    def compute(self):
        """
        Do the fitting
        :return:
        """
        self.fit()
        perr = np.sqrt(np.diag(self.pcov))
        print("Speed: " + str(self.popt[0]) + " +/- " + str(perr[0]))
        print("Day of the half point: " + str(self.popt[1]) + " +/- " + str(perr[1]))
        print("Total number: " + str(self.popt[2]) + " +/- " + str(perr[2]))

    def plot(self):
        """
        Plot sigmoid results
        :param popt: parameters array
        :return:
        """
        plt.title("Number of deaths since day 1 at " + self.state)
        plt.xlabel("Days since first case")
        plt.ylabel("Number")
        plt.scatter(self.xdata, self.ydata)
        x = np.linspace(2, 500, 5000)
        y = self.sigmoid(x, *self.popt)
        plt.plot(x, y)
        # plt.yscale('log')
        plt.xlim(45, 120)
        plt.ylim(0, max(y) * 1.1)
        plt.savefig(self.state + "_sigmoid.png", transperent=True)
        plt.show()


def main():
    state = "CA"
    timereader = TimeReader()
    stateData = timereader.getStateData(state)
    death = stateData['death']
    day = stateData['day']

    # Curve Fitting
    sigmoidfitter = SigmoidFitter(xdata=day, ydata=death, state=state)
    sigmoidfitter.compute()
    sigmoidfitter.plot()
    exit()


if __name__ == '__main__':
    main()
