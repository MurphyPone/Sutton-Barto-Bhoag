import numpy as np 
from visdom import Visdom 

viz = Visdom()

A_B = 0
A_E = 1 # end, back

def behavior_π():
    return np.random.binomial(1, 0.5)

def target_π():
    return A_B 

def play(): 
    trajectory = []
    while True: 
        action = behavior_π()
        trajectory.append(action)
        if action == A_E: 
            return 0, trajectory 
        else: 
            return 1, trajectory


def fig_5_4():
    runs = 10
    eps = 100000
    for run in range(runs):
        rewards = []
        for ep in range(0, eps):
            episodic_r, traj = play()
            if traj[-1] == A_E:
                ρ = 0
            else: 
                ρ = 1.0 / pow(0.5, len(traj))
            
            rewards.append(ρ * episodic_r)

        rewards = np.add.accumulate(rewards)
        estimations = np.asarray(rewards) / np.arange(1, eps + 1)
        
        viz.line(
            X=np.array([i for i in range(eps)]), 
            Y=np.array(estimations), 
            win="Estimated Reward Sampling", 
            name=f"Run {run}", 
            update="append",
            opts=dict(
                title="Ordinary Importance Sampling Sampling", 
                showlegend=True,
                layoutopts=dict(
                    plotly={
                        'xaxis': {'title': "episodes (log)", 'type': 'log'},
                        'yaxis': {'title': "v_π(s) with ordinary importance sampling"}
                    }
                )
            )
        )

if __name__ == '__main__':
    fig_5_4()