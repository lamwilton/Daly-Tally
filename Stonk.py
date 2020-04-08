import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
from scipy import interpolate
import datetime
import numpy as np
import math


def main():
    """
    # Interpolating COVID19 and Dow index and plot derivative
    :return:
    """
    # Read from data
    dowDf = pd.read_csv("^DJI.csv")
    covidDf = pd.read_csv("daily.csv")

    # Number of days
    day0 = datetime.date(2020, 1, 15)
    covidDf['dateChecked'] = pd.to_datetime(covidDf['dateChecked']).dt.date
    covidDf['day'] = (covidDf['dateChecked'] - day0).dt.days
    dowDf['Date'] = pd.to_datetime(dowDf['Date']).dt.date
    dowDf['Day'] = (dowDf['Date'] - day0).dt.days

    # Sum all states
    sumDf = ps.sqldf("SELECT day, SUM(positive) as cases FROM covidDf GROUP BY dateChecked")
    # Compute daily increases in cases, and daily rate of increase
    sumDf['dxdt'] = sumDf['cases'].diff()
    sumDf['rate'] = sumDf['dxdt'] / sumDf['cases']

    # Remove stock data before Covid19 appeared
    dowDf = dowDf[dowDf['Day'] > 48]

    # Interpolate using Cubic Splines
    covidInter = interpolate.CubicSpline(sumDf['day'], sumDf['cases']).derivative(1)
    dowInter = interpolate.CubicSpline(dowDf['Day'], dowDf['Close'])

    # Plot
    fig = plt.figure(facecolor='w', figsize=(10, 7))

    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_facecolor('#dddddd')
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.set_title("Daily growth rate of Corona cases")
    ax.set_xlabel("Days since first case")
    ax.set_ylabel("DOW Index/% growth rate * 100")
    ax.plot(sumDf['day'], sumDf['rate'] * 100000, 'b', label="COVID daily % growth rate")
    ax.plot(dowDf['Day'], dowInter(dowDf['Day']), 'r', label="DOW Jones")
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.savefig('DOW.png', transperent=True)
    plt.show()


if __name__ == '__main__':
    main()
    exit()
