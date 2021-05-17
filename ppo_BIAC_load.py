import gym

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym import wrappers
from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy
import matplotlib.pyplot as plt
import os

import pickle
import alr_envs


path = "./logs/ppo/ALRBallInACupSimpleDense-v0_1"

def make_env(rank, seed=0):
    def _init():
        env = gym.make('ALRBallInACupSimpleDense-v0')
        # env = wrappers.Monitor(env)#, path, force=True)
        return env

    return _init

n_cpu = 1
env = DummyVecEnv(env_fns=[make_env(i) for i in range(n_cpu)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

stats_path = os.path.join(path, "PPO.pkl")
env = DummyVecEnv(env_fns=[make_env(i) for i in range(n_cpu)])
env = VecNormalize.load(stats_path, env)#, norm_obs=True)#, norm_reward=True,clip_obs=10.)

model_path = os.path.join(path, "PPO.zip")
model = PPO.load(model_path)

obs = env.reset()

for i in range(200):
    action, _states = model.predict(obs, deterministic = True)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()   
 
