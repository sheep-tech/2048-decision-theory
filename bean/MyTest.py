# imports
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

# 2048 custom env object which inherites from OpenAI Gym's env methods
class My2048Env(Env):
    def __init__(self):
        # actions we can do: swipe up, down, left, right,
        self.action_space = Discrete(4)

        # # array
        # self.observation_space = Box(low=np.array([0]), high=np.array([100]))

        # set start board state with for loop
        # self.state

        # we might set the game moves length, i.e. stop after 256 steps to save memory
        # self.game_length = 256

    def step(self, action):
        # update board state
        # self.state +=

        # in case we adopt game_length, decrease it
        # self.game_length -= 1

        # if we get a single 2048 tile, we win(R=1), otherwise punish the user  agent(R=-1)

        pass

    def render(self):
        pass

    def reset(self):
        pass
