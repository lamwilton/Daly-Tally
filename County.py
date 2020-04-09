import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np


class County:
    def __init__(self):
        """
        https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/
        """
        self.population_df = pd.read_csv("covid_county_population_usafacts.csv")
        self.confirmed_df = pd.read_csv("covid_confirmed_usafacts.csv")
        self.death_df = pd.read_csv("covid_deaths_usafacts.csv")

    def cleaning(self):
        # Date format only works for Windows
        today = datetime.date(2020, 4, 8)
        todaystr = today.strftime("%#m/%#d/%y")

        # Remove unallocated counties, find days since first blood
        self.confirmed_df = self.confirmed_df[self.confirmed_df['countyFIPS'] != 0]
        self.confirmed_df['daysSinceFirst'] = (today - datetime.date(2020, 1, 22)).days - \
                                              self.confirmed_df.isin([0]).sum(axis=1) + 1
        # Remove all the previous date data, calculate cases per 100000 population
        self.confirmed_df = self.confirmed_df[['countyFIPS', 'County Name', 'State', 'stateFIPS', 'daysSinceFirst', todaystr]]
        self.confirmed_df['per1e5'] = self.confirmed_df[todaystr] / self.population_df['population'] * 100000

        self.death_df = self.death_df[self.death_df['countyFIPS'] != 0]
        self.death_df['daysSinceFirst'] = (today - datetime.date(2020, 1, 22)).days - \
                                              self.death_df.isin([0]).sum(axis=1) + 1
        self.death_df = self.death_df[['countyFIPS', 'County Name', 'State', 'stateFIPS', 'daysSinceFirst', todaystr]]
        self.death_df['per1e5'] = self.death_df[todaystr] / self.population_df['population'] * 100000

        self.population_df = self.population_df[self.population_df['countyFIPS'] != 0]
        print()


if __name__ == '__main__':
    county = County()
    county.cleaning()
