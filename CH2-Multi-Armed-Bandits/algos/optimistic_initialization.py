from math import sqrt, log 
import random
import numpy as np
from bandit import Bandit 
from visualize import *

# number of possible actions in in the k-armed bandit problem
k = 10
n_agents = 1000
α = 0.1 

# train the agent with optimal ε hyperparameter
def train(ε, init_value):
    bandit = [Bandit(k) for _ in range(n_agents)] 

    # track each of the agent's optimal, sub-optimal actions 1, 0 
    successes = np.array([0] * n_agents)

    # initialize action-value estimates and # times each action has been taken to some intial value
    Q = [[init_value] * k for _ in range(n_agents)]

    for t in range(1000):
        for agent in range(n_agents):
            # take randome action with p(ε), 
            # otherwise take action that maximises action-value estimates
            if random.random() < ε: 
                a = random.randint(0, k-1)
            else: 
                a = Q[agent].index(max(Q[agent]))

            r = bandit[agent].act(a)                # get a reward by taking action a against the bandit
            Q[agent][a] += α * (r - Q[agent][a])    # update stored values

            # store optimal/sub-optimal counts
            if a == bandit[agent].optimal_a:
                successes[agent] = 1  
            else: 
                successes[agent] = 0     

    # plot μ progress for all agents 
        new = True if t == 0 else False 
        plot(t, successes.mean(), '% Optimal Action: Optimistic Initialization', f"ε: {ε}, init: {init_value}", new)
          