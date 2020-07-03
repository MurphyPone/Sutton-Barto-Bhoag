import numpy as np
from visdom import Visdom

viz = Visdom()

def contour(data, title):
    viz.contour(
        X=np.array(data),
        win=title,
        opts=dict(
            title=title
        )
    )

def plot_V(V):
    contour(V, 'Values - Car Rental')

def plot_π(π):
    contour(π, 'Policy - Car Rental')
