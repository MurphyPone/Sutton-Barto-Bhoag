import random 

positions = [0] * 5 # A,B,C,D,E
GOAL = len(positions)

def is_done(index):
    # reached the end (E), or 
    return index == -1 or index == GOAL


def reward(index): 
    return 1 if index == GOAL else 0 # only get rewards if reached goal

def reset():
    return GOAL // 2 # start at the middle

def step(index):
    new_i = index + 1 if random.random() < 0.5 else index - 1
    r = reward(new_i)
    done = True if is_done(new_i) else False
    return new_i, r, done
