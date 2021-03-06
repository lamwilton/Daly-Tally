import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
import csv


class TimeReader:
    """
    Reading timeseries data of number of cases/deaths of COVID19
    """
    def __init__(self):
        """
        Data from https://covidtracking.com/api/v1/states/daily.csv
        """
        self.df = pd.read_csv("https://covidtracking.com/api/v1/states/daily.csv")
        self.sumdf = pd.DataFrame()
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

        # Compute sum of all states for USA
        df = self.df
        self.sumdf = ps.sqldf("SELECT dateChecked, SUM(death) FROM df GROUP BY dateChecked")
        self.sumdf['dateChecked'] = pd.to_datetime(self.sumdf['dateChecked']).dt.date
        self.sumdf['day'] = (self.sumdf['dateChecked'] - day0).dt.days

    def getStateData(self, state):
        """
        Get cases and deaths of state
        :param state: State input
        :return: Cases and deaths as dataframes
        """
        df = self.df
        stateData = ps.sqldf("SELECT * from df WHERE state = '" + state + "' ORDER BY dateChecked")
        return stateData

    def getsumDf(self):
        """
        Outputs national data
        :return:
        """
        return self.sumdf


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

        p0 = [1, 100, 50000]  # this is an mandatory initial guess
        self.popt, self.pcov = curve_fit(self.sigmoid, self.xdata, self.ydata, p0=p0, sigma=sigma, maxfev=1000000)

    def compute(self):
        """
        Do the fitting
        :return:
        """
        self.fit()
        perr = np.sqrt(np.diag(self.pcov))
        print("State: " + str(self.state))
        print("Speed: " + str(self.popt[0]) + " +/- " + str(perr[0]))
        print("Day of the half point: " + str(self.popt[1]) + " +/- " + str(perr[1]))
        print("Total number: " + str(self.popt[2]) + " +/- " + str(perr[2]))

        # Write to csv
        with open('output.csv', 'a', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow([self.state, self.popt, perr])

    def plot(self):
        """
        Plot sigmoid results
        :return:
        """
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, axisbelow=True)
        ax.set_facecolor('#dddddd')
        ax.set_title("Number of deaths since day 1 at " + self.state)
        ax.set_xlabel("Days since first case")
        ax.set_ylabel("Number")
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        ax.scatter(self.xdata, self.ydata)
        x = np.linspace(2, 500, 5000)
        y = self.sigmoid(x, *self.popt)
        ax.plot(x, y)

        ax.set_xlim(45, 120)
        ax.set_ylim(0, max(y) * 1.1)
        plt.savefig(self.state + "_sigmoid.png", transperent=True)
        
        ax.set_yscale('log')
        ax.set_ylim(1, max(y) * 1.1)
        plt.savefig(self.state + "_sigmoid_log.png", transperent=True)
        plt.show()


def main():
    """
    Main method
    :return:
    """
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
              "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
              "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
              "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
    states = []
    timereader = TimeReader()
    for state in states:
        stateData = timereader.getStateData(state)
        death = stateData['death']
        day = stateData['day']

        # Curve Fitting
        sigmoidfitter = SigmoidFitter(xdata=day, ydata=death, state=state)
        sigmoidfitter.compute()
        sigmoidfitter.plot()

    # National data
    sumdf = timereader.getsumDf()
    death = sumdf['SUM(death)']
    day = sumdf['day']
    # Curve Fitting
    sigmoidfitter = SigmoidFitter(xdata=day, ydata=death, state="US")
    sigmoidfitter.compute()
    sigmoidfitter.plot()
    exit()


if __name__ == '__main__':
    main()
