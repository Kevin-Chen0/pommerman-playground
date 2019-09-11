import random
from . import BaseAgent
from . import SimpleAgent


class DullAgent(BaseAgent):

    def __init__(self):
        BaseAgent.__init__(self)
        self.simple = SimpleAgent()

    def act(self, obs, action_space):
        step = int(obs['step_count'])
        if step > 100:
            return 5
        act = self.simple.act(obs, action_space)
        if act == 5:
            return random.choice(range(5))
        return act
