from math import sqrt, log 
import random
import numpy as np
from bandit import Bandit 
from visualize import *

# number of possible actions in in the k-armed bandit problem
k = 10
n_agents = 1000
α = 0.1

# train an agent with the given ε hyperparameter 
def train(c):
    bandit = [Bandit(k) for _ in range(n_agents)] 

    # track each of the agent's optimal, sub-optimal actions 1, 0 
    successes = np.array([0] * n_agents)

    # action-value estimates and # times each action has been taken 
    Q = [[0] * k for _ in range(n_agents)]
    N = [[0] * k for _ in range(n_agents)]

    for t in range(1000):
        for agent in range(n_agents):
            # take action with max estimated value, while also allowing for exploration
            func = [ Q[agent][i] + (c * sqrt(log(t + 1) / (N[agent][i] + 1))) for i in range(k) ]
            a = func.index(max(func))

            # get a reward by taking action a against the bandit
            r = bandit[agent].act(a)

            # update stored action-values
            N[agent][a] += 1
            Q[agent][a] += α * (r - Q[agent][a])


            # store optimal/sub-optimal counts
            if a == bandit[agent].optimal_a:
                successes[agent] = 1 
            else:
                successes[agent] = 0 
        
        # plot μ progress for all agents 
        new = True if t == 0 else False 
        plot(t, successes.mean(), '% Optimal Action: UCB', f"c: {c}", new)