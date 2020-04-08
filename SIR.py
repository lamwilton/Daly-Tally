import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class SIR:
    """
    SIR modeling class
    https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
    """
    def __init__(self, N=1000, I0=1, R0=0, beta=0.2, gamma=1./10, days=160):
        # Total population, N.
        self.N = N
        # Initial number of infected and recovered individuals, I0 and R0.
        self.I0, self.R0 = I0, R0
        # Everyone else, S0, is susceptible to infection initially.
        self.S0 = self.N - self.I0 - self.R0
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        self.beta, self.gamma = beta, gamma
        # A grid of time points (in days)
        self.t = np.linspace(0, days, days)

    def deriv(self, y, t, N, beta, gamma):
        """
        The SIR model differential equations.
        :param y:
        :param t:
        :param N:
        :param beta:
        :param gamma:
        :return: dSdt, dIdt, dRdt
        """
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def modeling(self):
        """
        Do the modeling SIR
        :return: S, I, R parameters
        """
        # Initial conditions vector
        y0 = self.S0, self.I0, self.R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(self.deriv, y0, self.t, args=(self.N, self.beta, self.gamma))
        S, I, R = ret.T
        return S, I, R

    def integrateFit(self, x, beta, gamma):
        """
        Fitting beta and gamma
        :param beta:
        :param gamma:
        :return:
        """
        # Initial conditions vector
        y0 = self.S0, self.I0, self.R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(self.deriv, y0, x, args=(self.N, beta, gamma))
        S, I, R = ret.T
        return S, I, R

    def plot(self, S, I, R):
        """
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        :param S:
        :param I:
        :param R:
        :return:
        """
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, axisbelow=True)
        ax.set_facecolor('#dddddd')
        ax.plot(self.t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(self.t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(self.t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Number (1000s)')
        ax.set_ylim(0, 1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()


def main():
    """
    Main method
    :return:
    """
    sir = SIR()
    S, I, R = sir.modeling()
    sir.plot(S, I, R)
    exit()


if __name__ == '__main__':
    main()