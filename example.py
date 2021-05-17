import gym
import numpy as np
import alr_envs

#env = gym.make('my_envs:BIAC-v0')
#env = gym.make('my_envs:Pusher-v0')
env = gym.make('alr_envs:ALRBallInACupSimpleDense-v0')

env.reset()
while True:
    action = np.random.randn(7)
    #a = np.zeros(7)
    #a = np.ones(5)
    #action[0] = 0
    #action[:] = -a
    #action[3] = -0.1
    env.render()
    env.step(action)

