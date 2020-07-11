from collections import deque 
import time 

from environment import *
from policy import *
from visualize import *

ε = 0.3
γ = 0.9

track_id = 2
max_episodes = 40000
vis_iter = 500

# create env 
env = Env(track_id=track_id)
behavior_π = BehaviorPolicy(ε)
target_π = TargetPolicy(env)

# create tables for Q function, Cum. weights, and # times each state has been visited
Q = np.zeros((len(env.track[0]), len(env.track), 5, 5, 3, 3) )
Q.fill(-200)
C = np.zeros((len(env.track[0]), len(env.track), 5, 5, 3, 3) )
visited = np.zeros((len(env.track[0]), len(env.track)))

# run episodes with given policy, return all s, a, r
def run_episode(policy, show=False):
    S, A, R = [], [], []

    s = env.reset()
    while True:
        S.append(tuple(s))

        if show: 
            a = policy.get_action(s) # only show the target π
        else: 
            if random.random() < 0.1: 
                a = np.array([0, 0])
            else: 
                a = policy.get_action(s, Q)
            
        s, r, done = env.step(a)  # done = terminal 

        A.append(tuple(a))
        R.append(r)

        visited[tuple(env.pos)] += 1

        if show:
            time.sleep(0.2)
            map(env.track, env.pos, track_id)

        if done:
            break
    
    return S, A, R 

# train it 
print('Training...', end='', flush=True)
ep_lens = []

for ep in range(max_episodes):
    # gen new episode using behavior policy
    S, A, R = run_episode(behavior_π)

    # store the len of the episode, updating the plot of episode * state visits occaisionally
    ep_lens.append(len(R))
    if ep % vis_iter == vis_iter -1: 
        plot_ep_len(ep_lens, track_id)
        freq_map(visited, track_id)

    G = 0
    W = 1
    for t in reversed(range(len(R))):
        s_a = S[t] + A[t]
        G = γ * G + R[t]
        C[s_a] += W
        Q[s_a] += (W / C[s_a]) * (G - Q[s_a])
        target_π.update_π(S[t], Q)
        if A[t] != tuple(target_π.get_action(np.array(S[t]))):
            break
        W *= 1 / behavior_π.get_prob(S[t], A[t], Q, ε)

print('Finished')
# demo the target π
for _ in range(10):
    run_episode(target_π, show=True)


