from math import sqrt 
from copy import deepcopy

from env import * 
from visualize import *

# true values from book
true_values = [1/6, 1/3, 1/2, 2/3, 5/6, 0]
γ = 0.1

def init_V():
    V = [0.5] * GOAL
    V.append(0) # V(done) = 0 <-- v[-1], V[GOAL]
    return V 

# return the Root Mean Square Error 
def RMS(x):
    n = len(x[:-1])
    error = 0
    for i in range(n):
        error += (true_values[i] - x[i])**2
    return sqrt(error / n)

def train_V():
    V = init_V()

    all_V = [true_values, deepcopy(V)] # init all V functions with true state values and init estimates
    plot_V(all_V)

    for ep in range(1, 101):
        # step through env performing TD(0) updates to the s-v estimates
        s = reset() 
        while True: 
            s2, r, done = step(s)
            V[s] += γ * (r + V[s2] - V[s])
            s = s2 

            if done: 
                break 

        # occaionally plot the updated Vs
        if ep == 1 or ep == 10 or ep == 100:
            all_V.append(deepcopy(V))
            plot_V(all_V)


def train_RMS(α):
    all_RMS = []

    for agent in range(100):
        V = init_V()
        agent_RMS = [RMS(V)]

        for ep in range(100):
            # step through env performing TD(0) updates to the s-v estimates
            s = reset() 
            while True: 
                s2, r, done = step(s)
                V[s] += α * (r + V[s2] - V[s])
                s = s2 

                if done: 
                    break 

            agent_RMS.append(RMS(V))
        all_RMS.append(agent_RMS)

    means = np.array(all_RMS).mean(axis=0)
    plot_RMS(means, α)

if __name__ == '__main__':
    train_V()
    train_RMS(0.15)
    train_RMS(0.1)
    train_RMS(0.05)

