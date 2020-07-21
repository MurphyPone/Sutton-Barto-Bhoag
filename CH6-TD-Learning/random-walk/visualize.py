import numpy as np
from visdom import Visdom
from env import *

viz = Visdom()

def plot_V(vals):
    ep_nums = [0, 1, 10, 100]
    legend = ['true values'] + [f'Ep {ep_nums[i]}' for i in range((len(vals)-1))]
    dash = np.array(['dash'] + ['solid'] * (len(vals)-1))
    title = 'Random Walk State Values'
    vals = np.array(vals).transpose()
    vals = vals[:-1] # last val is the terminal state, so we don't plot it 

    viz.line(
        X=np.arange(len(vals)),
        Y=np.array(vals),
        win=title,
        opts=dict(
            title=title,
            ytickmin=0,
            ytickmax=1,
            legend=legend,
            dash=dash
        )
    )

rms_win = None
rms_legend = []

def plot_RMS(errors, α):
    global rms_win, rms_legend

    title ='RMS Error'
    legend = f"α: {α}"
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
                legend=rms_legend
            )
        )