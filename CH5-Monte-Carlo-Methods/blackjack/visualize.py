from visdom import Visdom
import numpy as np

viz = Visdom()
win = None

def heatmap(data, title):
    viz.heatmap(
            X=np.flipud(data), 
            win=title, 
            opts=dict(
                title=title,
                columnnames=list(reversed(range(1,11))),
                rownames=list(reversed(range(12,22))),
                layoutopts=dict(
                    plotly={
                        'xaxis': {'title': "dealer showing"},
                        'yaxis': {'title': "agent sum"}
                    }
                )
            )
        )

def plot(eps, loss, title, name, update):
   viz.line(
        X=np.array([i for i in range(eps)]), 
        Y=np.array(loss), 
        win=title, 
        name=name, 
        update=update,
        opts=dict(
            title=title, 
            showlegend=True,
            layoutopts=dict(
                plotly={
                    'xaxis': {'title': "episodes (log)", 'type': 'log'},
                    'yaxis': {'title': "MSE"}
                }
            )
        )
    )


# def plot(x, y, title, name, new):
#     global win

#     if new:
#         if win != title:
#             win = viz.line(
#                 X=np.array([x]),
#                 Y=np.array([y]),
#                 win=title,
#                 name=name,
#                 opts=dict(
#                     title=title,
#                     showlegend=True,
#                     xtype='log',
#                     layoutopts=dict(
#                         plotly={
#                             'xaxis': {'title': "episodes (log)"},
#                             'yaxis': {'title': "MSE"}
#                         }
#                     )
#                 )
#             )
#         else:
#             win = viz.line(
#                 X=np.array([x]),
#                 Y=np.array([y]),
#                 win=title,
#                 name=name,
#                 update='new',
#                 opts=dict(
#                     title=title,
#                     showlegend=True,
#                     xtype='log',
#                     layoutopts=dict(
#                         plotly={
#                             'xaxis': {'title': "episodes (log)"},
#                             'yaxis': {'title': "MSE"}
#                         }
#                     )
#                 )
#             )
#     else:
#         win = viz.line(
#             X=np.array([x]),
#             Y=np.array([y]),
#             win=win,
#             name=name,
#             update='append'
#         )