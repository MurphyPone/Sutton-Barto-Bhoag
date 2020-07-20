from algos import ε_greedy, optimistic_initialization, ucb, gradient_bandit 

# train ε-greedy agents with different ε-values to verify optimal ε
epsilons = [0, 0.01, 0.1]
for ε in epsilons:
    ε_greedy.train(ε)

# UCB training with different c values 
c_vals = [1, 2, 5]
for c in c_vals:
    ucb.train(c)

# train ε-greedy agent with optimistic initial estimates and compare performance to std. ε-greedy
optimistic_initialization.train(ε=0, init_value=5)
optimistic_initialization.train(ε=0.1, init_value=0)
optimistic_initialization.train(ε=0.1, init_value=5)

# train gradient bandit agent with different α values 
alphas = [0.01, 0.1, 0.4]
for α in alphas:
    gradient_bandit.train(α)

