import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
from scipy import interpolate
import datetime


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

    # Join tables
    newDf = ps.sqldf("SELECT * FROM (SELECT day, cases FROM sumDf) NATURAL JOIN (SELECT Day, close FROM dowDf)")

    # Interpolate using Cubic Splines
    covidInter = interpolate.CubicSpline(newDf['day'], newDf['cases']).derivative(1)
    dowInter = interpolate.CubicSpline(newDf['day'], newDf['close'])

    # Plot
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_facecolor('#dddddd')
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.plot(newDf['day'], covidInter(newDf['day']), 'b')
    ax.plot(newDf['day'], dowInter(newDf['day']), 'r')
    plt.show()


if __name__ == '__main__':
    main()
    exit()
