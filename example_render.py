# render example 1
import gym
import numpy as np
env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"
test_env = gym.make(env_name)
test_env.render("human")
params = np.random.randn(15)
test_env.step(params)

# render example 2
import gym
import numpy as np
env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"
test_env = gym.make(env_name, render_mode="human")
params = np.random.randn(15)
test_env.step(params)

