from algos import ε_greedy #, optimistic_initialization, UCB, gradient_bandit


# train ε-greedy agents with different ε-values to verify optimal ε
epsilons = [0, 0.01, 0.1]

for ε in epsilons:
    ε_greedy.train(ε)

