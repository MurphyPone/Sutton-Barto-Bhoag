import numpy as np
from visdom import Visdom

viz = Visdom()
win = None

def plot(step, reward, title, data, new):
    global win

    if new:
        if win != title:
            win = viz.line(
                X=np.array([step]),
                Y=np.array([reward]),
                win=title,
                name=data,
                opts=dict(
                    title=title,
                    showlegend=True
                )
            )
        else:
            win = viz.line(
                X=np.array([step]),
                Y=np.array([reward]),
                win=title,
                name=data,
                update='new',
                opts=dict(
                    title=title
                )
            )
    else:
        win = viz.line(
            X=np.array([step]),
            Y=np.array([reward]),
            win=win,
            name=data,
            update='append'
        )