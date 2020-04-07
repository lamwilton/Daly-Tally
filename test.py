import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt


def readData():
    """
    Read NY times csv data
    :return: pandas dataframe df, states df of 50 states
    """
    df = pd.read_csv("us-states.csv")
    # Drop the FIPS column
    df = ps.sqldf(query="SELECT date,state,cases,deaths FROM df")
    states = ps.sqldf(query="SELECT DISTINCT state FROM df WHERE state NOT IN ('Puerto Rico', 'Virgin Islands', 'Guam', 'Northern Mariana Islands') ORDER BY state")
    return df, states


def getStateData(df, state):
    """
    Get cases and deaths of state
    :param df: Dataframe main
    :param state: State input
    :return: Cases and deaths as dataframes
    """
    ca = ps.sqldf(query="SELECT * FROM df WHERE state='" + state + "'")
    cases = ps.sqldf(query="SELECT cases FROM ca")
    deaths = ps.sqldf(query="SELECT deaths FROM ca")
    plt.plot(cases)
    plt.show()
    return cases, deaths


if __name__ == '__main__':
    df, states = readData()
    cases, deaths = getStateData(df, state='California')
    print(cases)
    print(deaths)