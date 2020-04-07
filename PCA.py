import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from ppca import PPCA


class PCA:
    """
    Main class

    Attributes: df  Main Dataframe from csv
    X  Data to analyze
    """
    def __init__(self):
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
        self.X = self.df2.values

        states = states.values.flatten()
        return states

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
        result = self.probPCA()
        # Plot
        plt.title("Probabilistic PCA of data of 2 dimensions")
        plt.scatter(result[:, 0], result[:, 1])
        for i, state in enumerate(states):
            plt.text(result[i, 0] + 0.5, result[i, 1] + 0.5, state)
        plt.savefig('PPCA.png', transperent=True)
        plt.show()


if __name__ == '__main__':
    pca = PCA()
    pca.mainRun()
    exit()