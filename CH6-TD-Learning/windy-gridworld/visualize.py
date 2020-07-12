import numpy as np
from visdom import Visdom

viz = Visdom()

all_eps = []
all_ts = []

def plot_t(ep, t, num_a):
    all_eps.append(ep)
    all_ts.append(t)
    viz.line(
        X=np.array(all_eps),
        Y=np.array(all_ts),
        win=f"Episode Len - {str(num_a)} Actions",
        opts=dict(
            title=f"Episode Len - {str(num_a)} Actions"
        )
    )

def map(env):
    grid = np.zeros((env.w, env.h))
    for i in range(len(env.wind)):
        grid[i:] = env.wind[i]
    grid[tuple(env.start_pos)] = 3
    grid[tuple(env.goal)] = 3
    grid[tuple(env.pos)] = 4
    grid = grid.transpose()
    viz.heatmap(
        X=grid,
        win="Windy Gridworld",
        opts=dict(
            title="Windy Gridworld"
        )
    )