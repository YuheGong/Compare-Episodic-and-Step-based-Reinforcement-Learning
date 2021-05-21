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
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class VevNormalizeCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(VevNormalizeCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        last_dist = np.mean([self.model.env.venv.envs[i].last_dist \
                             for i in range(len(self.model.env.venv.envs))
                             if self.model.env.venv.envs[i].last_dist != 0])
        last_dist_final = np.mean([self.model.env.venv.envs[i].last_dist_final \
                                   for i in range(len(self.model.env.venv.envs)) 
                                   if self.model.env.venv.envs[i].last_dist_final != 0])
        total_dist= np.mean([self.model.env.venv.envs[i].total_dist \
                             for i in range(len(self.model.env.venv.envs))])
        total_dist_final = np.mean([self.model.env.venv.envs[i].total_dist_final \
                                   for i in range(len(self.model.env.venv.envs))])
        min_dist = np.mean([self.model.env.venv.envs[i].min_dist \
                            for i in range(len(self.model.env.venv.envs))])
        min_dist_final = np.mean([self.model.env.venv.envs[i].min_dist_final \
                                  for i in range(len(self.model.env.venv.envs))])
        self.logger.record('reward/last_dist', last_dist)
        self.logger.record('reward/last_dist_final', last_dist_final)
        self.logger.record('reward/total_dist', total_dist)
        self.logger.record('reward/total_dist_final', total_dist_final)
        self.logger.record('reward/min_dist', min_dist)
        self.logger.record('reward/min_dist_final', min_dist_final)
        return True


class DummyCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(DummyCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        last_dist = np.mean([self.model.env.envs[i].last_dist \
                             for i in range(len(self.model.env.envs))
                             if self.model.env.envs[i].last_dist != 0])
        last_dist_final = np.mean([self.model.env.envs[i].last_dist_final \
                                   for i in range(len(self.model.env.envs))
                                   if self.model.env.envs[i].last_dist_final != 0])
        total_dist= np.mean([self.model.env.envs[i].total_dist \
                             for i in range(len(self.model.env.envs))])
        total_dist_final = np.mean([self.model.env.envs[i].total_dist_final \
                                   for i in range(len(self.model.env.envs))])
        min_dist = np.mean([self.model.env.envs[i].min_dist \
                            for i in range(len(self.model.env.envs))])
        min_dist_final = np.mean([self.model.env.envs[i].min_dist_final \
                                  for i in range(len(self.model.env.envs))])
        self.logger.record('reward/last_dist', last_dist)
        self.logger.record('reward/last_dist_final', last_dist_final)
        self.logger.record('reward/total_dist', total_dist)
        self.logger.record('reward/total_dist_final', total_dist_final)
        self.logger.record('reward/min_dist', min_dist)
        self.logger.record('reward/min_dist_final', min_dist_final)
        return True

class NormalCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(NormalCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        last_dist = self.model.env.last_dist
        last_dist_final = self.model.env.last_dist_final
        total_dist= self.model.env.total_dist
        total_dist_final = self.model.env.total_dist_final
        min_dist = self.model.env.min_dist
        min_dist_final = self.model.env.min_dist_final
        self.logger.record('reward/last_dist', last_dist)
        self.logger.record('reward/last_dist_final', last_dist_final)
        self.logger.record('reward/total_dist', total_dist)
        self.logger.record('reward/total_dist_final', total_dist_final)
        self.logger.record('reward/min_dist', min_dist)
        self.logger.record('reward/min_dist_final', min_dist_final)
        return True



if __name__ == "__main__":

    algorithm = 'ppo'

    def make_env(env_name,path, rank, seed=0):

        def _init():
            env = gym.make('alr_envs:ALRBallInACupSimpleDense-v0')
            env = wrappers.Monitor(env, path, force=True)
            return env

        return _init

    env_name = 'alr_envs:ALRBallInACupSimpleDense-v0'
    path = logging(env_name, algorithm)

    n_cpu = 4
    #env = DummyVecEnv(env_fns=[make_env(env_name, path, i) for i in range(n_cpu)])
    #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    env = gym.make(env_name)

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
                tensorboard_log= path,
                learning_rate=0.01,
                batch_size = 50,
                n_steps=2000)
    model.learn(total_timesteps=int(5e6), callback=DummyCallback())  # , callback=TensorboardCallback())

    # save the model
    model_path = os.path.join(path, "PPO.zip")
    model.save(model_path)
    #stats_path = os.path.join(path, "PPO.pkl")
    #env.save(stats_path)
