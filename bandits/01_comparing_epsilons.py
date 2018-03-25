import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt


# m is true mean
# set instance variables mean and N to 0
# N is number of times we pull/play
# mean instance variable is our estimate of the bandit's mean

class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0


# pull function simulates pulling the arm
    def pull(self):
        return np.random.randn() + self.m


# x is the latest sample received from the bandit
# function to return the cumulative average after every play
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x  # update mean equation


def run_experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]


    # results in an array data, size N
    data = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        # if p is less than epsilon, choose bandit at random
        if p < eps:
            j = np.random.choice(3)
        # otherwise choose bandit with best current sample mean
        else:
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
    # for epsilon = 10%, 5%, 1%
    c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
    c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)


    # log scale plot to see fluctuations in earlier rounds more clearly
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')

    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()


# did better with optimistic initial values


