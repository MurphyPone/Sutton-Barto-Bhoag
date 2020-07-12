import random 
import numpy as np 
import time 

from environment import Env 
from visualize import * 

max_episodes = 2000
ε = 0.1
α = 0.5 
γ = 1 

# action sets
four_a = [[0, 1], [0, -1], [1, 0], [-1, 0]]
eight_a = [[0, 1], [0, -1], [1, 0], [-1, 0], [-1, 1], [1, 1], [1, -1]]
nine_a = [[0, 1], [0, -1], [1, 0], [-1, 0], [-1, 1], [1, 1], [1, -1], [0, 0]]

actions = four_a # eight_a, nine_a
use_stochastic_wind = False

env = Env(actions, use_stochastic_wind)

Q = np.zeros((env.w, env.h, len(actions)))
def π(s): 
    if random.random() < ε:         # ε greedy policy based on Q 
        return random.randint(0, len(actions) - 1)
    else: 
        return np.argmax(Q[tuple(s)])

# update Q function using SARSA
def update_Q(s, a, r, s2, a2):
    s_a = tuple(s) + tuple([a])
    s2_a2 = tuple(s2) + tuple([a2])
    Q[s_a] += α * (r + (γ * Q[s2_a2] - γ * Q[s_a]))

# train π using SARSA to create state-value estimates
def train(): 
    for ep in range(max_episodes):
        s = env.reset()
        a = π(s)
        t = 0
        while True: 
            t += 1 
            s2, r, done = env.step(a)
            a2 = π(s2)
            update_Q(s, a, r, s2, a2)

            s = s2 
            a =  a2 
            if done: 
                break 

        plot_t(ep, t, len(actions))

# display an agent traversing the env with the deterministic ε-greedy policy
def demo():
    s = env.reset()
    a = np.argmax(Q[tuple(s)])

    while True: 
            s2, r, done = env.step(a)
            a2 = np.argmax(Q[tuple(s2)]) # no need to explore anymore
            s = s2 
            a =  a2 
            map(env)
            time.sleep(0.2)
            if done: 
                break 


if __name__ == '__main__':
    train() 
    demo()