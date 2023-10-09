import gym
import gym_maze
import numpy
import random
import time


env = gym.make('maze-sample-5x5-v0')
print(env.reset())

action = 0
print('step', env.step(action))