from math import *
import numpy as np
from visualize import *

dim = 20
γ = 0.9

def poisson(λ, n):
    return pow(λ, n) * np.exp(-λ) / factorial(n)

def expected_return(s, a, V):
    ret = 0.0

    # each future state will have negative reward, so we just apply it at the beginning
    ret -= 2 * abs(a) 

    # loop through each combination of rentals and returns
    for num_rentals_1 in range(11):
        for num_rentals_2 in range(11):
            # probability of the number of rentals occurring
            p_rentals_1 = poisson(3, num_rentals_1)
            p_rentals_2 = poisson(4, num_rentals_2)

            # number of cars left at each location after moving car overnight
            num_cars_1 = np.clip(s[0] - a, 0, dim)
            num_cars_2 = np.clip(s[1] + a, 0, dim)

            # number of rentals allowed based on number of cars actually at each location
            actual_rentals_1 = min(num_cars_1, num_rentals_1)
            actual_rentals_2 = min(num_cars_2, num_rentals_2)

            # price of all rentals
            reward = (actual_rentals_1 + actual_rentals_2) * 10


            """ constant returns/day -> considerably shorter run time """
            p_total = p_rentals_1 * p_rentals_2 
            num_cars_1 = np.clip(num_cars_1 - actual_rentals_1 + 3, 0, dim)
            num_cars_2 = np.clip(num_cars_1 - actual_rentals_1 + 2, 0, dim)
            ret += p_total * (reward + (γ * V[num_cars_1][num_cars_2]))
            
            """ poisson returns/day -> considerably longer run time """
            # for num_returns_1 in range(11):
            #     p_returns_1 = poisson(3, num_returns_1)
            #
            #     for num_returns_2 in range(11):
            #         p_returns_2 = poisson(2, num_returns_2)
            #
            #         p_total = p_rentals_1 * p_rentals_2 * p_returns_1 * p_returns_2
            #
            #         num_cars_1 = np.clip(num_cars_1 - actual_rentals_1 + num_returns_1, 0, dim)
            #         num_cars_2 = np.clip(num_cars_2 - actual_rentals_2 + num_returns_2, 0, dim)
            #
            #         ret += p_total * (reward + (0.9 * V[num_cars_2][num_cars_1]))
    return ret 

def policy_evaluation(states, π, V):
    θ = 1
    while True: 
        Δ = 0

        # set values of each state to the expected retrns follow the current policy at those states 
        for x,y in states:
            v = V[x][y]
            a = π[x][y]
            V[x][y] = expected_return((x, y), a, V)

            # update the max change in values for this iteration
            Δ = max(Δ, abs(v - V[x][y]))

    plot_V(V)

    # stop updating values if max change is small enough 
    if Δ < θ:
        return V

def policy_iteration(states, π, V):
    stable = True 

    # update the policy to selecte the action that maximizes the expected reward for each state
    for x,y in states: 
        old_a = π[x][y]

        max_a = old_a 
        max_return = expected_return((x,y), old_a, V)
        for a in range(-5, 6):
            # some actions can't physically be performed, so we exclude them for speed
            if (a >= 0 and x >= a) or (a < 0 and y >= abs(a)):
                cur_return = expected_return((x,y), a, V)
                if max_return < cur_return:
                    max_a = a
                    max_return = cur_return
        
        π[x][y] = max_a
        plot_π(π)

        # if the new policy matches the old policy, then it's optimal and we can stop iterating
        if old_a != π[x][y]:
            stable = False
    
    return π, stable

if __name__ == '__main__': 
    
    states = []                 # init the states
    for x in range(dim + 1):
        for y in range(dim +1):
            states.append((x,y))

    # init the value and policy to 0s
    V = [[0] * (dim+1) for _ in range(dim + 1)]
    π = [[0] * (dim+1) for _ in range(dim + 1)]

    # run till optimal policy found
    epoch = 0
    while True: 
        epoch += 1
        # print(f"Epoch {epoch}, π evaluation...")
        # V = policy_evaluation(states, π, V)
        # print("π iteration...")
        # π, stable = policy_iteration(states, π, V)
        # print("DONE")

        # if stable:
        #     break

        print('Epoch %d: Policy Evaluation...' % epoch, end='', flush=True)
        V = policy_evaluation(states, π, V)

        print('Policy Iteration...', end='', flush=True)
        π, stable = policy_iteration(states, π, V)
        print('DONE')

        if stable:
            break

