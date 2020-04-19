# COVID-19 #hackforhope hackathon code

Devpost link: https://devpost.com/software/daly-tally/  
More information: Daly-tally.pdf  

Neural.ipynb

Multilayer perceptron Neural net for county level medicare and COVID-19 data
Jupyter Notebook for running NN for predicting the number of cases (deaths) for county level data. The data is cleaned and generated from data of 1 million medicare providers in US and the COVID county level data 

PCA.py

PCA of COVID data vs DALY scores
Running PCA on the DALY scores and COVID number of cases, deaths etc on the 50 states and generate a 2D plot, where the states are separated based on political affiliation

GrowthRate.py

Computing exponential growth rate of number of cases and visualize, comparing with DOW index
Like the bank interest rate, I visualized the daily growth rates of COVID number of cases of the whole country and all 50 states. For comparison I plotted the growth rate with the DOW index

TimeReader.py

Reading timeseries data of number of cases/deaths of COVID19, and fitting the data to the sigmoid function and plot it
