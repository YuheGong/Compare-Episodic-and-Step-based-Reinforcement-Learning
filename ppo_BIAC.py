import gym
import numpy as np
import numpy as np
import torch as th
from gym import wrappers
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.ppo import MlpPolicy
import os
import pickle
import tensorflow as tf
import alr_envs
from utils.logger import logging

if __name__ == "__main__":

    def make_env(env_name,path, rank, seed=0):

        def _init():
            env = gym.make('alr_envs:ALRBallInACupSimpleDense-v0')
            env = wrappers.Monitor(env, path, force=True)
            return env

        return _init

    env_name = 'alr_envs:ALRBallInACupSimpleDense-v0'
    algorithm = 'ppo'
    path = logging(env_name, algorithm)

    n_cpu = 1
    env = DummyVecEnv(env_fns=[make_env(env_name, path, i) for i in range(n_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    #env.render("human")
    ALGOS = {
        'a2c': A2C,
        'dqn': DQN,
        'ddpg': DDPG,
        'her': HER,
        'sac': SAC,
        'ppo': PPO,
        'td3': TD3
    }
    ALGO = ALGOS[algorithm]

    model = ALGO(MlpPolicy, env, verbose=1,
                # policy_kwargs=policy_kwargs,
                tensorboard_log= path,
                learning_rate=0.00001,
                n_steps=2048)
    model.learn(total_timesteps=int(1.2e6))  # , callback=TensorboardCallback())

    # save the model
    model_path = os.path.join(path, "PPO.zip")
    model.save(model_path)
    stats_path = os.path.join(path, "PPO.pkl")
    env.save(stats_path)
