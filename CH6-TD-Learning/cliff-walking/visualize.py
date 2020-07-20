import numpy as np 
from visdom import Visdom 

viz = Visdom()
all_r = []

legend = ['SARSA', 'Q-Learning', 'Expected SARSA']

np.seterr('raise')
def plot(r, new):
    means = np.mean(r, axis=0)

    if new:
        all_r.append(means)
    else: 
        all_r[len(all_r)-1] = means

    viz.line(
        Y=np.array(all_r).transpose(),
        win='Episodic Returns',
        opts=dict(
            title='Episodic Returns',
            legend=legend[:len(all_r)]
        )
    )