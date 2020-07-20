import random 
import numpy as np 
from hyperparams import *

class Agent():
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.Q = np.zeros((self.env.w, self.env.h, 4))

    def get_action(self, s):
        if random.random() < ε:
            return random.randint(0, 3)
        else: 
            return np.argmax(self.Q[tuple(s)])

class QLearning(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def update_Q(self, s, a, r, s2, a2):
        s_a = tuple(s) + tuple([a])
        self.Q[s_a] += α * (r + (γ * np.max(self.Q[tuple(s2)])) - self.Q[s_a])

class reg_SARSA(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def update_Q(self, s, a, r, s2, a2):
        s_a = tuple(s) + tuple([a])
        s2_a2 = tuple(s2) + tuple([a2])
        self.Q[s_a] += α * (r + (γ * self.Q[s2_a2]) - self.Q[s_a])

class Expected_SARSA(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)
    
    def expected_next_Q(self, s2):
        E = 0
        for a2 in range(4):
            s2_a2 = tuple(s2) + tuple([a2])
        
            if a2 == np.argmax(self.Q[tuple(s2)]):
                prob = 1 - ε + (ε / 4) 
            else: 
                prob = ε / 4
        
            E += prob * self.Q[s2_a2]
        
        return E 

    def update_Q(self, s, a, r, s2, a2):
        s_a = tuple(s) + tuple([a])
        s2_a2 = tuple(s2) + tuple([a2])
        self.Q[s_a] += α * (r + (γ * self.expected_next_Q(s2)) - self.Q[s_a])



