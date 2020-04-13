import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from ppca import PPCA


class PCA:
    """
    PCA of COVID data vs DALY scores
    """

    def __init__(self):
        """
        Data from James
        """
        self.df = pd.read_csv("Data.csv")
        self.df2 = pd.DataFrame()
        self.X = np.empty_like

    def readData(self):
        """
        Processing df and extract states
        :return: states abbr array
        """
        POPULATION_UNIT = 1000000
        states = pd.DataFrame(self.df[['state']])

        # Make new Dataframe, Select all columns except first four
        self.df2 = self.df.loc[:, "DaysSinceStayatHomeOrder":]

        # Divide all DALY scores, Corona stuff and Allriskfactors by population per 1000000
        self.df2.loc[:, "15-49yearsAllcauses":"Smoking"] = \
            self.df2.loc[:, "15-49yearsAllcauses":"Smoking"].div(self.df2["TotalPop2018"], axis=0) * POPULATION_UNIT
        self.df2.loc[:, "positive":"onVentilatorCumulative"] = \
            self.df2.loc[:, "positive":"onVentilatorCumulative"].div(self.df2["TotalPop2018"], axis=0) * POPULATION_UNIT
        self.df2.loc[:, "Allriskfactors"] = self.df2.loc[:, "Allriskfactors"].div(self.df2["TotalPop2018"], axis=0)
        # Remove 'pending', population columns
        self.df2 = self.df2.drop(columns=["pending"])
        self.df2 = self.df2.drop(columns=["TotalPop2018", "Log10Pop"])
        '''
        self.df2 = self.df2[['DaysSinceStayatHomeOrder', 'DaysSinceFirstPositive',
                              'DaysSinceTestStart', 'positive', 'negative', 'death', 'hospitalized',
                              'totaltests', 'recovered', 'inIcuCumulative', 'onVentilatorCumulative',
                              'Noaccesstohandwashingfacility', 'Smoking',
                              'PercentUrbanPop', 'Density(P/mi2)', 'DaysSinceInfection',
                              'LandArea(mi2)']]
        '''
        self.X = self.df2.values

        states = states.values.flatten()
        return states

    def getdf(self):
        """
        Return processed dataframe
        :return: df2
        """
        return self.df2

    def probPCA(self):
        """
        Do probabilistic PCA
        :return: result 2d array X
        """
        # Normalizing X
        sc = StandardScaler()
        X_normalized = sc.fit_transform(self.X)

        ppca = PPCA()
        ppca.fit(data=X_normalized, d=2)
        result = ppca.transform()
        return result

    def mainRun(self):
        states = self.readData()

        # Define state colors
        stateColor = {'AL': 0, 'AK': 0, 'AZ': 0, 'AR': 0, 'CA': 2, 'CO': 1, 'CT': 2, 'DE': 2, 'DC': 2, 'FL': 1, 'GA': 0,
                      'HI': 2, 'ID': 0, 'IL': 2, 'IN': 1, 'IA': 1, 'KS': 0, 'KY': 0, 'LA': 0, 'ME': 2, 'MD': 2, 'MA': 2,
                      'MI': 1, 'MN': 2, 'MS': 0, 'MO': 0, 'MT': 0, 'NE': 0, 'NV': 1, 'NH': 2, 'NJ': 2, 'NM': 1, 'NY': 2,
                      'NC': 1, 'ND': 0, 'OH': 1, 'OK': 0, 'OR': 2, 'PA': 1, 'RI': 2, 'SC': 0, 'SD': 0, 'TN': 0, 'TX': 0,
                      'UT': 0, 'VT': 2, 'VA': 1, 'WA': 2, 'WV': 0, 'WI': 1, 'WY': 0}
        cdict = {0: 'red', 1: 'purple', 2: 'blue'}
        result = self.probPCA()

        # Plot each state
        plt.figure(facecolor='w', figsize=(10, 7))
        plt.title("Probabilistic PCA of data of 2 dimensions")
        for i, state in enumerate(states):
            plt.scatter(result[i, 0], result[i, 1], c=cdict[stateColor[state]])
            plt.text(result[i, 0] + 0.2, result[i, 1] + 0.2, state)
        plt.savefig('PPCA.png', transperent=True)
        plt.show()


if __name__ == '__main__':
    pca = PCA()
    pca.mainRun()
    exit()
