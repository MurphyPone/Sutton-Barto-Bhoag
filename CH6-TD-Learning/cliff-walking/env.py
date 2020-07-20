import numpy as np 

class Env():
    def __init__(self):
        self.w = 12
        self.h = 4

        self.actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.actions = [np.array(dir) for dir in self.actions]

        self.start_pos = np.array([0, 0])
        self.goal = np.array([self.w-1, 0])

    def reset(self):
        self.pos = self.start_pos 
        return self.pos

    def at_goal(self):
        return tuple(self.pos) == tuple(self.goal)

    def off_cliff(self):
        return self.pos[1] == 0 and self.pos[0] > 0 and self.pos[0] < (self.w - 1)

    def step(self, a_ind):
        self.pos = np.clip(self.pos + self.actions[a_ind], 0, [self.w-1, self.h-1])
        done = self.at_goal() or self.off_cliff()
        if self.off_cliff():
            self.reset()
            r = -100
        else: 
            r = -1
        return self.pos, r, done