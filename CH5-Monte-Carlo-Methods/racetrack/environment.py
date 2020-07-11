import random 
import numpy as np

MIN_VEL, MAX_VEL = 0, 4

class Env(): 
    def __init__(self, track_id=1):
        # 'X' = wall, 
        # '.' = track, 
        # '-' = start
        # '|' = finish
        if track_id == 1:
            self.track = [
                'XXXXXXXXXXXXXXXXXX',                       # X = wall
                'XXXX.............|',                       # . = track
                'XXX..............|',                       # - = start
                'XXX..............|',                       # | = finish
                'XX...............|',
                'XX...............|',
                'XX...............|',
                'X..........XXXXXXX',
                'X.........XXXXXXXX',
                'X.........XXXXXXXX',
                'X.........XXXXXXXX',
                'X.........XXXXXXXX',
                'X.........XXXXXXXX',
                'X.........XXXXXXXX',
                'X.........XXXXXXXX',
                'XX........XXXXXXXX',
                'XX........XXXXXXXX',
                'XX........XXXXXXXX',
                'XX........XXXXXXXX',
                'XX........XXXXXXXX',
                'XX........XXXXXXXX',
                'XX........XXXXXXXX',
                'XX........XXXXXXXX',
                'XXX.......XXXXXXXX',
                'XXX.......XXXXXXXX',
                'XXX.......XXXXXXXX',
                'XXX.......XXXXXXXX',
                'XXX.......XXXXXXXX',
                'XXX.......XXXXXXXX',
                'XXX.......XXXXXXXX',
                'XXXX......XXXXXXXX',
                'XXXX......XXXXXXXX',
                'XXXX------XXXXXXXX',                       # bottom of track -> y = 0 (even though that's not how it's stored in the array)
            ]
        elif track_id == 2:
            self.track = [
                'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
                'XXXXXXXXXXXXXXXXX...............|',
                'XXXXXXXXXXXXXX..................|',
                'XXXXXXXXXXXXX...................|',
                'XXXXXXXXXXXX....................|',
                'XXXXXXXXXXXX....................|',
                'XXXXXXXXXXXX....................|',
                'XXXXXXXXXXXX....................|',
                'XXXXXXXXXXXXX...................|',
                'XXXXXXXXXXXXXX..................|',
                'XXXXXXXXXXXXXXX................XX',
                'XXXXXXXXXXXXXXX.............XXXXX',
                'XXXXXXXXXXXXXXX............XXXXXX',
                'XXXXXXXXXXXXXXX..........XXXXXXXX',
                'XXXXXXXXXXXXXXX.........XXXXXXXXX',
                'XXXXXXXXXXXXXX..........XXXXXXXXX',
                'XXXXXXXXXXXX............XXXXXXXXX',
                'XXXXXXXXXXX.............XXXXXXXXX',
                'XXXXXXXXXX..............XXXXXXXXX',
                'XXXXXXXXX...............XXXXXXXXX',
                'XXXXXXXX................XXXXXXXXX',
                'XXXXXXX.................XXXXXXXXX',
                'XXXXXX..................XXXXXXXXX',
                'XXXXX...................XXXXXXXXX',
                'XXXX....................XXXXXXXXX',
                'XXX.....................XXXXXXXXX',
                'XX......................XXXXXXXXX',
                'X.......................XXXXXXXXX',
                'X.......................XXXXXXXXX',
                'X-----------------------XXXXXXXXX',
            ]
        elif track_id == 3:
            self.track = [
                'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
                'XXXXXXXXXXXXXXXXX...............|',
                'XXXXXXXXXXXXXX..................|',
                'XXXXXXXXXXXXX...................|',
                'XXXXXXXXXXXX....................|',
                'XXXXXXXXXXXX....................|',
                'XXXXXXXXXXXX....................|',
                'XXXXXXXXXXXX....................|',
                'XXXXXXXXXXXXX...................|',
                'XXXXXXXXXXX.....................|',
                'XXXXXXXXXX.....................XX',
                'XXXXXXX.....................XXXXX',
                'XXXXX......................XXXXXX',
                'XXXX...........XXXX........XXXXXX',
                'XXX.........XXXXXXX........XXXXXX',
                'XXX.........XXXXXXX........XXXXXX',
                'XXX.........XXXXXXX........XXXXXX',
                'XXX.........XXXXXXX........XXXXXX',
                'XXX.........XXXXXXX........XXXXXX',
                'XXX.........XXXXXXX........XXXXXX',
                'XXX.........XXXXXXX........XXXXXX',
                'XXX.........XXXXXXX........XXXXXX',
                'XXX........................XXXXXX',
                'XXX.........................XXXXX',
                'XXX...........................XXX',
                'XXX---------------------------XXX',
            ]

        # Keep track of valid start positions
        self.start_positions = []
        for i in range(len(self.track[0])):
            if self.track[len(self.track)-1][i] == '-':
                self.start_positions.append(i)
    
    def get_state(self):
        return np.concatenate((self.pos, self.vel))

    # move agent to random starting pos and reset its 2D velocity
    def reset(self):
        self.pos = np.array([random.choice(self.start_positions), 0])
        self.vel = np.array([0, 0])
        return self.get_state()

    def step(self, a):
        # update vel
        self.vel = np.clip(self.vel + a, MIN_VEL, MAX_VEL)

        # move accordingly
        self.pos = np.clip(self.pos + self.vel, [0, 0], [len(self.track[0])-1, len(self.track)-1])

        # reset if OOB w/o ending episode (only end if agent reaches finish line)
        cur_pos = self.track[ len(self.track) - 1 - self.pos[1] ][ self.pos[0] ]
        if cur_pos == 'X':
            self.reset()
        elif cur_pos == '|':
            return None, - 1, True          # s', r, terminal, shouldn't reward be nonzero?

        return self.get_state(), -1, False


        