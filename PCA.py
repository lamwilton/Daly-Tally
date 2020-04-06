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
        states = pd.DataFrame(self.df[['state']])

        # Make new Dataframe, Select all columns except first four
        self.df2 = self.df.iloc[:, 4:]
        self.X = self.df2.values

        # Divide all DALY scores by population
        self.X[:, 13:65] = np.dot(self.X[:, 13:65].T, np.diag(1 / self.X[:, 12])).T
        # Remove 'pending' column, too many NaNs
        self.X = np.delete(self.X, 5, axis=1)
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
        plt.title("Probabilistic PCA of data of 2 dimensions, DALY scores population weighted")
        plt.scatter(result[:, 0], result[:, 1])
        for i, state in enumerate(states):
            plt.text(result[i, 0] + 0.5, result[i, 1] + 0.5, state)
        plt.savefig('PPCA.png', transperent=True)
        plt.show()


if __name__ == '__main__':
    pca = PCA()
    pca.mainRun()
    exit()