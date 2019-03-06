from DPMethods import *
from imports import *

env = gym.make("FrozenLake8x8-v0")
agent = PolicyIteration(env.env)
agent.play(plot=True)


