import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ppca import PPCA


def readData():
    """
    Read csv data
    :return: states abbr, data X
    """
    df = pd.read_csv("Data.csv")
    states = pd.DataFrame(df[['state']])

    # Select all columns except first four
    X = df.iloc[:, 4:].values
    states = states.values.flatten()
    return states, X


if __name__ == '__main__':
    states, X = readData()

    # Normalizing X
    sc = StandardScaler()
    X_normalized = sc.fit_transform(X)

    # Probabilistic PCA, good for dealing with missing data
    ppca = PPCA()
    ppca.fit(data=X_normalized, d=2, verbose=True)
    result = ppca.transform()

    # Plot
    plt.scatter(result[:,0], result[:,1])
    for i, state in enumerate(states):
        plt.text(result[i,0]+0.5, result[i,1]+0.5, state)
    plt.savefig('PPCA.png', transperent=True)
    plt.show()
