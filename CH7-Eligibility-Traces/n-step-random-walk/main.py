import numpy as np
from tqdm import tqdm
# import matplotlib
# import matplotlib.pyplot as plt
from visualize import *

N_STATES = 19
γ = 1           
STATES = np.arange(1, N_STATES + 1)
START_S = 10            # start from middle 
END_S = [0, N_STATES + 1]
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0 # from Bellman
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0 # terminal states on either end 

def get_action(s):
    if np.random.binomial(1, 0.5) == 1:
        return s + 1
    else: 
        return s - 1

def get_reward(s2):
    if s2 == 0:
        return -1
    elif s2 == 20:
        return 1
    else:
        return 0

def temporal_diff(V, n, α):
    s = START_S
    states = [s]
    rewards = [0]
    t = 0
    T = float('inf')
    
    while True:
        t += 1

        if t < T:
            s2 = get_action(s)
            r = get_reward(s2)
            states.append(s2) 
            rewards.append(r)

            if s2 in END_S:
                T = t

        update_time = t - n
        if update_time >= 0:
            G = 0.0 # returns

            # calc corresponding rewards
            for time in range(update_time + 1, min(T, update_time + n) + 1):
                G += pow(γ, t - update_time - 1) * rewards[time]
            if update_time + n <= T:
                G += pow(γ,n) * V[states[(update_time + n)]]
            
            s_to_update = states[update_time]
            if not s_to_update in END_S:
                V[s_to_update] += α * (G - V[s_to_update])
        
        if update_time == T - 1:
            break
        s = s2
        
def figure_7_2():
    steps = np.power(2, np.arange(0, 3))
    alphas = np.arange(0, 1.1, 0.1)
    n_eps = 10 #10
    n_runs = 10 # 100

    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(0, n_runs)):
        for s, step in enumerate(steps):
            for a, alpha in enumerate(alphas):
                v = np.zeros(N_STATES + 2)
                for ep in range(0, n_eps):
                    temporal_diff(v, step, alpha) 
                    errors[s, a] += np.sqrt(np.sum(np.power(v - TRUE_VALUE, 2)) / N_STATES)

        
    errors /= n_eps * n_runs
    for i in range(0, len(steps)):
        print(errors[i:], len(errors[i:]), steps[i]) 
        plot_RMS(errors[i:], steps[i])


    # for i in range(0, len(steps)):
    #     plt.plot(alphas, errors[i, :], label='n = %d' % (steps[i]))
    # plt.xlabel('alpha')
    # plt.ylabel('RMS error')
    # plt.ylim([0.25, 0.55])
    # plt.legend()
    # plt.show()        

figure_7_2()