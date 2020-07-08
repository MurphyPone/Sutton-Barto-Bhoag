import numpy as np 
# from visualize import *

HIT = 0 
STAND = 1 # hit or stand
actions = [HIT, STAND] # TODO deprecate these to just one-hots

#### agent's policy denoted: π
π = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    π[i] = HIT 
π[20], π[21] = HIT, HIT

# agent's target policy
def target_π(agent_usable_ace, agent_sum, dealer_showing):
    return π[agent_sum]

# agent's behavior policy
def behavior_π(agent_usable_ace, dealer_showing):
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
    card = np.random(1, 14)
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
            action = target_π(agent_usable_ace, agent_sum, dealer_card1)
        
        # track actions for behavior sampling
        agent_trajectory.append([agent_usable_ace, agent_sum, dealer_card1], action)

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
    
    return ordinary_sampling