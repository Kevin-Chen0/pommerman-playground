from numpy import random
from . import BaseAgent
from . import RandomAgent
from . import SimpleAgent


class HybridAgent(BaseAgent):

    def __init__(self, eps=0.5):
        BaseAgent.__init__(self)
        self.random = RandomAgent()
        self.simple = SimpleAgent()
        self.eps = eps

    def act(self, obs, action_space):
        choice = random.choice(2, p=[self.eps, 1-self.eps])
        if choice == 0:
            return self.random.act(obs, action_space)
        else:
            return self.simple.act(obs, action_space)
