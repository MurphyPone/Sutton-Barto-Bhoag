from math import sqrt, log 
import random
import numpy as np
from bandit import Bandit 
from visualize import *

# number of possible actions in in the k-armed bandit problem
k = 10
n_agents = 1000

# softmax function to approximate exact gradient ascent
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# policy for choosing and evaluating actions
class Policy():
    def __init__(self, k):
        self.probs = [0] * k  # probabilities of receiving a given task

    # sample from the available actions
    def get_action(self, H):
        self.probs = softmax(H)
        return np.random.choice(k, 1, p=self.probs)[0]

    # get the probability of selecting a given action
    def get_prob(self, a):
        return self.probs[a]

# train an agent with the given ε hyperparameter 
def train(α):
    bandit = [Bandit(k) for _ in range(n_agents)] 

    # track each of the agent's optimal, sub-optimal actions 1, 0 
    successes = np.array([0] * n_agents)

    # initialize action-value estimates and # times each action has been taken to 0
    H = [[0] * k for _ in range(n_agents)]

    # total rewards
    r_total = [0 for _ in range(n_agents)]

    # create a policy
    policy = [Policy(k) for _ in range(n_agents)]

    for t in range(1000):
        for agent in range(n_agents):
            a = policy[agent].get_action(H[agent])  # take action by sampling the policy
            r = bandit[agent].act(a)                # get a reward by taking action a against the bandit
            r_total[agent] += r 

            # update preferences based on the approximate gradient of the expected reward    
            for i in range(k):
                if i == a: 
                    H[agent][i] += α * (r - (r_total[agent] / (t + 1))) * (1 - policy[agent].get_prob(i))
                else: 
                    H[agent][i] -= α * (r - (r_total[agent] / (t + 1))) * policy[agent].get_prob(i)

            # store optimal/sub-optimal counts
            if a == bandit[agent].optimal_a:
                successes[agent] = 1 
            else:
                successes[agent] = 0 
        
        # plot μ progress for all agents 
        new = True if t == 0 else False 
        plot(t, successes.mean(), '% Optimal Action: Gradient Bandit', f"α: {α}", new)