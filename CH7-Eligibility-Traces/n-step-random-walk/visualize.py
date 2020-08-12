import numpy as np
from visdom import Visdom

viz = Visdom()

rms_win = None
rms_legend = []

def plot_RMS(errors, n):
    global rms_win, rms_legend

    title ='RMS Error'
    legend = f"n-steps: {n}"
    rms_legend.append(legend)

    if rms_win is None:
        rms_win = viz.line(
            X=np.arange(len(errors)),
            Y=errors,
            win=title,
            name=legend,
            opts=dict(
                title=title,
                legend=rms_legend
            )
        )
    else: 
        rms_win = viz.line(
            X=np.arange(len(errors)),
            Y=errors,
            win=rms_win,
            name=legend,
            update='append',
            opts=dict(
                title=title,
                # legend=rms_legend
            )
        )