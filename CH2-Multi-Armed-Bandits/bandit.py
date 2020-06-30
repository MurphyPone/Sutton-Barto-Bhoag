import random 
import numpy as np

class Distribution():
    def __init__(self, μ, σ):
        self.μ, self.σ = μ, σ 
    
    # reward function samples from the normal distribution using μ, σ
    def sample(self):
        return np.random.normal(loc=self.μ, scale=self.σ)

    def __str__(self):
        return f"({self.μ}, {self.σ})"

class Bandit():
    def __init__(self, k):
        # Create k distributions with random μ in [-5, 5], σ = 1
        self.distributions = [Distribution(random.randint(-5, 5), 1) for _ in range(k)]

        # Find the distribution with highest μ 
        self.optimal_a = 0
        for i in range(k):
            pdf = self.distributions[i]
            if pdf.μ > self.distributions[self.optimal_a].μ:
                self.optimal_a = i 

    # Sample from the reward distribution corresponding to the agent's action
    def act(self, a):
        return self.distributions[a].sample()