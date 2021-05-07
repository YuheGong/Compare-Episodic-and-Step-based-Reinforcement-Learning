import gym
import numpy as np
import numpy as np
import torch as th

from gym import wrappers
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import os
import pickle
import tensorflow as tf
import alr_envs


if __name__ == "__main__":

    path = "./log/ppo"
    def make_env(rank, seed=0):

        def _init():
            env = gym.make('ALRBallInACupSimple-v0')
            #env = wrappers.Monitor(env)#, path, force=True)
            return env

        return _init

    n_cpu = 1
    env = DummyVecEnv(env_fns=[make_env(i) for i in range(n_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(MlpPolicy, env, verbose=1,
                # policy_kwargs=policy_kwargs,
                tensorboard_log= path,
                learning_rate=0.0001,
                n_steps=2048)  ######200)
    model.learn(total_timesteps=int(1.2e7))  # , callback=TensorboardCallback())

    # save the model
    log_dir = path
    model_path = os.path.join(log_dir, "PPO.zip")
    model.save(model_path)
    stats_path = os.path.join(log_dir, "PPO.pkl")
    env.save(stats_path)
