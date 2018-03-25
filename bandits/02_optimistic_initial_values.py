import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

from comparing_epsilons import run_experiment as run_experiment_eps


# m is true mean
# set instance variables mean and N to 0
# N is number of times we pull/play
# mean instance variable is our estimate of the bandit's mean

class Bandit:
    def __init__(self, m, upper_limit):
        self.m = m
        self.mean = upper_limit
        self.N = 1


# pull function simulates pulling the arm
    def pull(self):
        return np.random.randn() + self.m


# x is the latest sample received from the bandit
# function to return the cumulative average after every play
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x  # update mean equation


def run_experiment(m1, m2, m3, N, upper_limit=10):
    bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]


    # results in an array data, size N
    data = np.empty(N)

    for i in range(N):
        # optimistic initial values
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)


    # plot moving average ctr
    # comparing plots for different epsilon
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.mean)

    return cumulative_average

if __name__ == '__main__':
    c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
    oiv = run_experiment(1.0, 2.0, 3.0, 100000)


    # log scale plot to see fluctuations in earlier rounds more clearly
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(oiv, label='optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()


    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(oiv, label='optimistic')
    plt.legend()
    plt.show()


# code compares the epsilon greedy strategy with epsilon 10% optimistic initial value strategy
# did better with optimistic
