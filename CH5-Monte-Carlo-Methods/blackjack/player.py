import numpy as np 
from tqdm import tqdm
from visualize import *
from visdom import Visdom
viz = Visdom()

HIT = 0 
STAND = 1 # hit or stand
ACTIONS = [HIT, STAND] # TODO deprecate these to just one-hots

#### agent's policy denoted: π
π = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    π[i] = HIT 
π[20], π[21] = HIT, HIT

# agent's target policy
def target_π(agent_usable_ace, agent_sum, dealer_showing):
    return π[agent_sum]

# agent's behavior policy
def behavior_π(agent_usable_ace, agent_sum, dealer_showing):
    if np.random.binomial(1, 0.5) == 1:
        return STAND
    return HIT

#### dealer's policy denoted: ω
ω = np.zeros(22)
for i in range(12, 17):
    ω[i] = HIT 

for i in range(17, 22):
    ω[i] = STAND 

def get_card():
    card = np.random.randint(1, 14)
    return min(card, 10)     # face card value is 10

# used for aces
def card_value(card):
    return 11 if card == 1 else card 

def play(π, init_s=None, init_a=None):
    agent_sum = 0
    agent_trajectory = []
    agent_usable_ace = False

    dealer_card1, dealer_card2 = 0, 0 

    
    if init_s is None: # gen new random state
        while agent_sum < 12:
            card = get_card()
            agent_sum += card_value(card)

            if agent_sum > 21:
                assert agent_sum == 22 # check to make sure he's holding two aces if he's at 22
                agent_sum -= 10 
            else:
                agent_usable_ace |= (1 == card) # TODO
        
        dealer_card1, dealer_card2 = get_card(), get_card()

    else: # else take the initial state 
        agent_usable_ace, agent_sum, dealer_card1 = init_s 
        dealer_card2 = get_card()

    state = [agent_usable_ace, agent_sum, dealer_card1]

    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    dealer_usable_ace  = 1 in (dealer_card1, dealer_card2)

    if dealer_sum > 21: 
        assert dealer_sum == 22
        dealer_sum -= 10
    assert dealer_sum <= 21 
    assert agent_sum <= 21 

    # agent goes first
    while True:
        if init_a is not None:
            action = init_a
            init_a = None 
        else: 
            action = π(agent_usable_ace, agent_sum, dealer_card1)
        
        # track actions for behavior sampling
        agent_trajectory.append([(agent_usable_ace, agent_sum, dealer_card1), action])

        if action == STAND:
            break # end the agent's turn

        card = get_card()
        ace_count = int(agent_usable_ace) # track # aces since usable ace is effectively a bool
        if card == 1:
            ace_count += 1
        agent_sum += card_value(card)
        while agent_sum > 21 and ace_count:
            agent_sum -= 10
            ace_count -= 1
        if agent_sum > 21: # bust after converting Aces to 1s
            return state, -1, agent_trajectory
        assert agent_sum <= 21 # TODO this is redundant I think
        agent_usable_ace = (ace_count == 1)

    # dealer goes second
    while True:
        action = ω[dealer_sum]
        if action == STAND:
            break 

        card = get_card()
        ace_count = int(dealer_usable_ace)
        if card == 1:
            ace_count += 1
        dealer_sum += card_value(card)
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        
        if dealer_sum > 21: # dealer busts
            return state, 1, agent_trajectory
        dealer_usable_ace = (ace_count == 1)

    # determine winner if neither bust
    assert agent_sum <= 21 and dealer_sum <= 21
    if agent_sum > dealer_sum:
        return state, 1, agent_trajectory
    elif agent_sum == dealer_sum:
        return state, 0, agent_trajectory
    else: 
        return state, -1, agent_trajectory

def monte_carlo_on_policy(eps):
    states_usable_ace = np.zeros((10,10))
    states_usable_ace_count = np.ones((10, 10)) # init to 1 to prevent div/0
    states_no_usable_ace = np.zeros((10,10))
    states_no_usable_ace_count = np.ones((10, 10))

    for i in tqdm(range(0, eps)):
        _, episodic_r, agent_trajectory = play(behavior_π)
        for(usable_ace, agent_sum, dealer_card), action in agent_trajectory:
            agent_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[agent_sum, dealer_card] += 1
                states_usable_ace[agent_sum, dealer_card] += episodic_r
            else:
                states_no_usable_ace_count[agent_sum, dealer_card] += 1
                states_no_usable_ace[agent_sum, dealer_card] += episodic_r
    return states_usable_ace / states_no_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count

# MC with exploring start
def monte_carlo_ε(eps):
    # state action vals for agent_sum, deealer_card, usable_ace, action
    sa_vals = np.zeros((10, 10, 2, 2)) 
    sa_pair_count = np.ones((10, 10, 2, 2))

    # override 
    def behavior_π(usable_ace, agent_sum, dealer_showing):
        usable_ace = int(usable_ace)
        agent_sum -= 12 
        dealer_showing -=1
        vals = sa_vals[agent_sum, dealer_showing, usable_ace, :] / sa_pair_count[agent_sum, dealer_showing, usable_ace, :]
        
        return np.random.choice([action for action, val in enumerate(vals) if val == np.max(vals) ])

    for ep in tqdm(range(eps)):
        # randomly gen a sa pair
        init_s = [bool(np.random.choice([0, 1])), 
                        np.random.choice(range(12, 22)),
                        np.random.choice(range(1, 11))]
        init_a = np.random.choice(ACTIONS)
        current_π = behavior_π if ep else target_π
        _, episodic_r, trajectory = play(current_π, init_s=init_s, init_a=init_a)
        first_visit_check = set()
        
        for (usable_ace, agent_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            agent_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, agent_sum, dealer_card, action)
            
            if state_action in first_visit_check:
                continue

            first_visit_check.add(state_action)
            sa_vals[agent_sum, dealer_card, usable_ace, action] += episodic_r
            sa_pair_count[agent_sum, dealer_card, usable_ace, action] += 1
    
    return sa_vals / sa_pair_count

def monte_carlo_off_policy(eps):
    init_s = [True, 13, 2]

    rhos = [] # TODO
    returns = []

    for i in range(0, eps):
        _, episodic_r, agent_trajectory = play(behavior_π, init_s=init_s)

        # Calc importance ratio
        num = 1.0
        den = 1.0 
        for(usable_ace, agent_sum, dealer_card), action in agent_trajectory:
            if action == target_π(usable_ace, agent_sum, dealer_card):
                den *= 0.5
            else: 
                num = 0.0
                break
        ρ = num / den 
        rhos.append(ρ)
        returns.append(episodic_r)

    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    weighted_returns = rhos * returns
    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)

    ordinary_sampling = weighted_returns / np.arange(1, eps + 1) # plus one to avoid div/0

    # TODO
    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)
    
    return ordinary_sampling, weighted_sampling


## Plot results of the above MC methods

def fig_5_1():
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1, states_usable_ace_2,
                states_no_usable_ace_1, states_no_usable_ace_2]
    
    titles = ['usable A - 10,000 eps', 'usable A - 500,000 eps',
                'No usable A - 10,000 eps', 'No usable A - 500,000 eps']
    for state, title in zip(states, titles):
        heatmap(state, title)

def fig_5_2():
    sa_vals = monte_carlo_ε(500000)
    state_val_usable_ace = np.max(sa_vals[:, :, 1, :], axis=-1)
    state_val_no_usable_ace = np.max(sa_vals[:, :, 0, :], axis=-1)

    # optimal policies for ace/no ace states
    π_usable_ace = np.argmax(sa_vals[:, :, 1, :], axis=-1)
    π_no_usable_ace = np.argmax(sa_vals[:, :, 0, :], axis=-1)

    states = [π_usable_ace, state_val_usable_ace, 
                π_no_usable_ace, state_val_no_usable_ace]

    
    titles = ['π* w/ usable Ace', 'v* w/ usable Ace',
                'π* w/o usable Ace', 'v* w/o usable Ace']

    for state, title in zip(states, titles):
        heatmap(state, title)

def fig_5_3():
    true_val = -0.27726
    eps = 10000
    runs = 100
    err_ordinary, err_weighted = np.zeros(eps), np.zeros(eps)

    for i in tqdm(range(0, runs)):
        ordinary_sampling, weighted_sampling = monte_carlo_off_policy(eps)
        # calc MSE
        err_ordinary += np.power(ordinary_sampling - true_val, 2)
        err_weighted += np.power(weighted_sampling - true_val, 2)
    
    err_ordinary /= runs
    err_weighted /= runs

    plot(eps, err_ordinary, "Importance Sampling", "Ordinary", "append")
    plot(eps, err_weighted, "Importance Sampling", "Weighted", "append")

    # viz.line(
    #     X=np.array([i for i in range(eps)]), 
    #     Y=np.array(err_ordinary), 
    #     win="Importance Sampling", 
    #     name="Ordinary", 
    #     opts=dict(
    #         title="Importance Sampling", 
    #         showlegend=True,
    #         layoutopts=dict(
    #             plotly={
    #                 'xaxis': {'title': "episodes (log)", 'type': 'log'},
    #                 'yaxis': {'title': "MSE"}
    #             }
    #         )
    #     )
    # )

    # viz.line(
    #     X=np.array([i for i in range(eps)]), 
    #     Y=np.array(err_weighted), 
    #     win="Importance Sampling", 
    #     name="Weighted", 
    #     update='append',
    #     opts=dict( 
    #         showlegend=True,
    #         layoutopts=dict(
    #             plotly={
    #                 'xaxis': {'title': "episodes (log)", 'type': 'log'},                    
    #                 'yaxis': {'title': "MSE"}
    #             }
    #         )
    #     )
    # )
    
    # plot(eps, err_weighted, 'Importance Sampling', "Weighted")

    
#    def plot(x, y, title, name, new):



if __name__ == '__main__':
    # fig_5_1()
    # fig_5_2()
    fig_5_3()