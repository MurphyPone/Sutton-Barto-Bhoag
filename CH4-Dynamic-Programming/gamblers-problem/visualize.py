import numpy as np 
from visdom import Visdom

viz = Visdom()

def plot(S, data, title):
    viz.line(
        X=np.array(S),
        Y=np.array(data),
        win=title,
        opts=dict(
            title=title
        )
    )

def plot_V(S, V):
    plot(S, V[1:-1], "Values - Gambler")

def plot_π(S, π):
    plot(S, π[1:-1], "Policy - Gambler")