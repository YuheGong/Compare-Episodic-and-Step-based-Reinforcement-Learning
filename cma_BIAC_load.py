import cma
import gym
import numpy as np

from alr_envs.utils.mp_env_async_sampler import AlrMpEnvSampler
from alr_envs.mujoco.ball_in_a_cup import ball_in_a_cup

if __name__ == "__main__":

    algo = np.load("logs/cma/ALRBallInACupSimpleDMP-v0_1/algo_mean.npy")
    env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"
    test_env = gym.make(env_name)
    test_env.render("human")
    test_env.step(algo)

