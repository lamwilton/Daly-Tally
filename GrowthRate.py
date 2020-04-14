import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
from scipy import interpolate
import datetime
import numpy as np
from sys import exit


class GrowthRate:
    """
    Computing exponential growth rate of number of cases and visualize, comparing with DOW index
    """
    def __init__(self):
        """
        Read from data
        https://covidtracking.com/api/v1/states/daily.csv
        https://finance.yahoo.com/quote/%5EDJI/history/
        """
        self.dowDf = pd.read_csv("^DJI.csv")
        self.covidDf = pd.read_csv("daily.csv")
        self.states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
                       "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
                       "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
                       "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

    def cleaning(self):
        """
        Compute Number of days
        :return:
        """
        day0 = datetime.date(2020, 1, 15)
        self.covidDf['dateChecked'] = pd.to_datetime(self.covidDf['dateChecked']).dt.date
        self.covidDf['day'] = (self.covidDf['dateChecked'] - day0).dt.days
        self.dowDf['Date'] = pd.to_datetime(self.dowDf['Date']).dt.date
        self.dowDf['Day'] = (self.dowDf['Date'] - day0).dt.days

    def national(self):
        """
        Interpolating Dow index and plot with COVID data
        :return:
        """
        # Sum all states
        covidDf = self.covidDf
        sumDf = ps.sqldf("SELECT day, SUM(positive) as cases FROM covidDf GROUP BY dateChecked")
        # Compute daily increases in cases
        sumDf['dxdt'] = sumDf['cases'].diff()
        # Compute "daily interest rate" percentage change
        sumDf['rate'] = sumDf['cases'].pct_change().fillna(0)

        # Remove DOW data before Covid19 appeared
        self.dowDf = self.dowDf[self.dowDf['Day'] > 43]

        # Interpolate DOW using Cubic Splines
        dowInter = interpolate.CubicSpline(self.dowDf['Day'], self.dowDf['Close'])

        # Plot
        fig = plt.figure(facecolor='w', figsize=(10, 7))
        ax = fig.add_subplot(111, axisbelow=True)
        ax.set_facecolor('#dddddd')
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        ax.set_title("Daily growth rate of Corona cases")
        ax.set_xlabel("Days since first case")
        ax.set_ylabel("DOW Index/% growth rate * 50")
        ax.plot(sumDf['day'], sumDf['rate'] * 50000, 'b', label="COVID daily % growth rate")
        ax.plot(self.dowDf['Day'], dowInter(self.dowDf['Day']), 'r', label="DOW Jones")
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        plt.savefig('DOW.png', transperent=True)
        plt.show()

    def computeState(self, state):
        """
        Get cases and deaths of state, then interest rates
        :param state: State input
        :return: Cases and deaths as dataframes
        """
        covidDf = self.covidDf
        stateDf = ps.sqldf("SELECT * from covidDf WHERE state = '" + state + "' ORDER BY dateChecked")

        # Compute percentage changes and fill NaNs
        stateDf['rateIncrease'] = stateDf['positive'].pct_change().fillna(0)
        stateDf['deathrateIncrease'] = stateDf['death'].pct_change().fillna(0)
        return stateDf

    def plotState(self, state, stateDf):
        """
        Plot state graph
        :param state: What state
        :param stateDf: Dataframe for the state
        :return:
        """
        fig = plt.figure(facecolor='w', figsize=(10, 7))

        ax = fig.add_subplot(111, axisbelow=True)
        ax.set_facecolor('#dddddd')
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        ax.set_title("Daily growth rate of Corona cases in " + state)
        ax.set_xlabel("Days since first case")
        ax.set_ylabel("% growth rate")
        ax.plot(stateDf['day'], stateDf['rateIncrease'] * 100, 'b', label="COVID daily % growth rate")
        plt.savefig(state + '_rate.png', transperent=True)
        plt.show()

    def allStates(self):
        """
        Compute growth rates of all states
        :return:
        """
        rateTable, deathTable = pd.DataFrame(), pd.DataFrame
        for i in range(len(self.states)):
            # Compute rates
            stateDf = self.computeState(self.states[i])
            rate = stateDf[['date', 'rateIncrease']]
            deathrate = stateDf[['date', 'deathrateIncrease']]

            # Add result to rate tables
            if i <= 0:
                table = rate
                deathTable = deathrate
            else:
                table = pd.merge(table, rate, how='inner', on='date')
                deathTable = pd.merge(deathTable, deathrate, how='inner', on='date')
        arr = np.array(table.iloc[:, 1:])
        deatharr = np.array(deathTable.iloc[:, 1:])
        return arr, deatharr

    def pseudoColorPlot(self, arr):
        """
        Plot 2D pseudocolor of array
        :param arr: Data np array
        :return:
        """
        fig = plt.figure(facecolor='w', figsize=(10, 7))
        ax = fig.add_subplot(111, axisbelow=True)

        # Set title and axis
        ax.set_title("Daily % growth rate of Corona cases in US")
        ax.set_xlabel("Days since Mar 7 2020")
        ax.set_yticks(np.arange(len(self.states)) + 0.5)
        ax.set_yticklabels(self.states)
        ax.invert_yaxis()

        # Plot heatmap
        c = ax.pcolor(arr.T * 100, cmap='Blues', vmin=0, vmax=100)
        fig.colorbar(c, ax=ax)
        plt.savefig('US_rate.png', transperent=True)
        plt.show()


def main():
    """
    Main method
    :return:
    """
    growth = GrowthRate()
    growth.cleaning()
    growth.national()
    stateDf = growth.computeState("NY")
    growth.plotState("NY", stateDf)
    arr, deatharr = growth.allStates()
    growth.pseudoColorPlot(arr)
    exit()


if __name__ == '__main__':
    main()
