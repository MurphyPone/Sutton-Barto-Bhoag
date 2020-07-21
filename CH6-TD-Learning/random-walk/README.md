### Example 6.2: Random Walk 

In this example we empirically compare the prediction abilities of TD(0) and constant-α MC applied to the small Markov process shown in Figure 6.5. All episodes start in the center state, C, and proceed either left or right by one state on each step, with equal probability. This behavior is presumably due to the combined effect of a fixed policy and an environment’s state-transition probabilities, but we do not care which; we are concerned only with predicting returns however they are generated. Episodes terminate either on the extreme left or the extreme right. When an episode terminates on the right a reward of +1 occurs; all other rewards are zero. For example, a typical walk might consist of the following state-andreward sequence: C, 0, B, 0, C, 0, D, 0, E, 1. Because this task is undiscounted and episodic, the true value of each state is the probability of terminating on the right if starting from that state. Thus, the true value of the center state is vπ(C) = 0.5. The true values of all the states, A through E, are 

`1/6`,`2/6`, `3/6`, `4/6`, `5/6`,

In all cases the approximate value function is initialized to the intermediate value V (s) = 0.5, for all s. The TD method is consistently better than the MC method on this task over this number of episodes.

