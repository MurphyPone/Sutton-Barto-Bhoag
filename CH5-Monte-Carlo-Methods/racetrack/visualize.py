import numpy as np
from visdom import Visdom

viz = Visdom()

# convert ascii track to #'s and flip track vertically so we don't have to plot in reverse order
def track_to_nums(track):
    nums = [ [] for _ in range(len(track))] 

    for i in range(len(track)):
        row = track[i]
        for char in row:
            if char == 'X':
                nums[len(track)-1-i].append(0)
            elif char == '.':
                nums[len(track)-1-i].append(1)
            elif char == '|':
                nums[len(track)-1-i].append(2)
            else: # char == '-':
                nums[len(track)-1-i].append(3)
    
    return nums

def map(track, pos, id): 
    track = track_to_nums(track)
    track[pos[1]][pos[0]] = 4

    viz.heatmap(
        X=np.array(track),
        win=f'Racetrack {id}',
        opts=dict(
            title=f'Racetrack: {id}'
        )
    )

def freq_map(visited, id):
    viz.heatmap(
        X=visited.transpose(),
        win=f'Visted{id}',
        opts=dict(
            title=f'Frequency Visited: {id}',
            colormap='hot'
        )
    )

def plot_ep_len(ep_lens, id):
    viz.line(
        X=np.arange(len(ep_lens)),
        Y=np.array(ep_lens),
        win=f'Episode Lengths: {id}',
        opts=dict(
            title=f'Episode Lengths: {id}'
        )
    )