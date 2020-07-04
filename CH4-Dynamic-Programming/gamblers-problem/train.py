from math import * 
import numpy as np 
from visualize import * 

p_h = 0.4               # probability of getting heads
θ = 1e-10               # convergence factor for value iteration
γ = 1                   # discount (no discount)

# initialize the states as an array of his possible capital to stake
def init_S():
    return [i for i in range(1, 100)]

def init_V():
    V = [0] * 101
    V[100] = 1
    return V

def expected_return(s, a, V):
    ret = 0.0

    # s_{t+1} is either a winning or losing state
    # the reward is 0 at every state except in one of the terminal states
    #   so we can just use (γ * V(s')) instead of (r + γ*v(s'))
    ret += p_h * (γ * V[s + a])
    ret += (1 - p_h) * (γ * V[s - a]) 
    return ret

def value_iteration(S, V):
    while True: 
        Δ = 0

        for s in S:
            v = V[s]

            A = [i for i in range(min(s, 100-s) + 1)]

            # set value of given state equal to highest expected return from all possible actions in that state
            max_ret = 0 
            for a in A:
                cur_ret = expected_return(s, a, V)
                max_ret = cur_ret if cur_ret > max_ret else max_ret
            V[s] = max_ret

            # update the maximum change in value for this iter
            Δ = max(Δ, abs(v - V[s]))

        # stop updating when Δ < θ
        if Δ < θ:
            plot_V(S, V)
            return V

def make_policy(S, V):
    π = [0] * (len(S) + 2)

    # set the policy to maximize the value at each state (duh)
    # settle ties arbitrarily
    for s in S: 
        A = [i for i in range(min(s, 100-s) + 1)]

        max_ret = 0
        for a in A:
            cur_ret = expected_return(s, a, V)
            if cur_ret > max_ret:
                max_ret = cur_ret
                π[s] = a
    
    plot_π(S, π)
    return π


if __name__ == '__main__':
    S = init_S()
    V = init_V()
    V = value_iteration(S, V)
    π = make_policy(S, V)

