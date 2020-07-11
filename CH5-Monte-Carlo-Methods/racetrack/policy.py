import random 
import numpy as np

def get_max_a(s, Q):
    s = tuple(s)

    max_a = (0, 0)
    max_q = Q[s + max_a]

    for x_vel in range(-1, 2): # vel range [0, 4]
        for y_vel in range(-1, 2):
            a = (x_vel, y_vel)
            q = Q[s + a]
            if q > max_q:
                max_q = q
                max_a = a 
    
    return np.array(max_a)

class TargetPolicy():
    def __init__(self, env):
        track = env.track
        self.π = np.zeros((len(track[0]), len(track), 5, 5, 2)).astype(int)

    def update_π(self, s, Q):
        self.π[s] = get_max_a(s, Q)

    def get_action(self, s):
        return self.π[tuple(s)]

class BehaviorPolicy():
    def __init__(self, ε):
        self.ε = ε

    def get_action(self, s, Q):
        if random.random() < self.ε:
            return np.random.randint(low=-1, high=2, size=2)
        
        return get_max_a(s, Q)
    
    def get_prob(self, s, a, Q, ε):
        if tuple(get_max_a(np.array(s), Q)) == a:
            return 1 - ε + (ε / 9)
        return ε / 9