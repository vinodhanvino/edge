

class customeEnvironment:
    def __init__(self, obstacles):
        self.start = 0
        self.obstacles =  obstacles
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, array):
        array = array[-1]
        if array[-1] > 2.2:
            reward = 0.5
            done = False
        elif array[-1] == 2.2:
            reward = 0.1
            done = False
        elif array[-1] < 2.2:
            reward = -1
            done = False
        else:
            reward = 0
            done = False
        return reward, done

