import random 
import numpy as np 
from bandit import Bandit 
from visualize import * 

k = 10 
n_agents = 1000 

def train(ε):
    bandit = [Bandit(k) for _ in range(n_agents)] 

    # track each of the agent's optimal, sub-optimal actions 1, 0 
    successes = np.array([0] * n_agents)

    # action-value estimates and # times each action has been taken 
    Q = [[0] * k for _ in range(n_agents)]
    N = [[0] * k for _ in range(n_agents)]

    for t in range(1000):
        for agent in range(n_agents):
            if random.random() < ε: 
                a = random.randint(0, k - 1)        # explore
            else:
                a = Q[agent].index(max(Q[agent]))   # exploit

            r = bandit[agent].act(a)                # get reward from action

            N[agent][a] += 1
            # incremenal running rewards for that action
            Q[agent][a] += (1 / N[agent][a]) * (r - Q[agent][a]) 

            # store optimal/sub-optimal counts
            if a == bandit[agent].optimal_a:
                successes[agent] = 1  
            else: 
                successes[agent] = 0                                                   

        # plot μ progress for all agents 
        new = True if t == 0 else False 
        plot(t, successes.mean(), '% Optimal Action: ε-greedy', f"ε: {ε}", new)