import os
import alr_envs
import cma
import dill
import numpy as np
from cma.bbobbenchmarks import nfreefunclasses
import gym


import cw2.cluster_work
import cw2.cw_data.cw_pd_logger
import cw2.experiment
import os
import random

from cw2 import cluster_work, cw_error, experiment
from cw2.cw_data import cw_logging, cw_pd_logger
import numpy as np
import torch as th

from gym import wrappers
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import os
import pickle
import tensorflow as tf
from utils.callback import ALRBallInACupCallback,DMbicCallback
from utils.custom import CustomActorCriticPolicy
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.ppo import MlpPolicy


def make_env(env_name, rank):
    def _init():
        env = gym.make(env_name)
        # env = wrappers.Monitor(env, path, force=True)
        return env

    return _init

class CWCMA(cw2.experiment.AbstractIterativeExperiment):
    def __init__(self):
        super().__init__()
        self.env = None
        self.algorithm = None


    def initialize(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:

        pass



    def iterate(self, config: dict, rep: int, n: int) -> dict:
        env = DummyVecEnv(env_fns=[make_env(config.params.env_name, i) for i in range(config.params.env_num)])
        env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

        test_env = DummyVecEnv(env_fns=[make_env(config.params.env_name, i) for i in range(1)])
        test_env = VecNormalize(test_env, training=False, norm_obs=True, norm_reward=False)

        ALGOS = {
            'a2c': A2C,
            'dqn': DQN,
            'ddpg': DDPG,
            'her': HER,
            'sac': SAC,
            'ppo': PPO,
            'td3': TD3
        }
        ALGO = ALGOS[config.params.algorithm]

        POLICY = {
            'MlpPolicy': MlpPolicy,
        }

        path = config.path

        model = ALGO("MlpPolicy", env, verbose=1, create_eval_env=True,
                     # model = ALGO(MlpPolicy, env, verbose=1, create_eval_env=True,
                     tensorboard_log=path,
                     seed=3,
                     learning_rate=config.params.learning_rate,
                     batch_size=config.params.batch_size,
                     n_steps=config.params.n_steps)

        eval_callback = EvalCallback(test_env, best_model_save_path=path, n_eval_episodes=10,
                                     log_path=path, eval_freq=500,
                                     deterministic=False, render=False)

        model.learn(total_timesteps=int(config.params.total_timesteps), callback=eval_callback)

    def save_state(self, config: dict, rep: int, n: int) -> None:
        #if n % 50 == 0:
        #    f_name = os.path.join(
        #        config.rep_log_paths[rep], 'optimizer.pkl')
        #    with open(f_name, 'wb') as f:
        #        dill.dump(self.optimizer, f)
        pass

    def finalize(self, surrender = None, crash: bool = False):
        pass

    def restore_state(self):
        pass


if __name__ == "__main__":
    cw = cw2.cluster_work.ClusterWork(CWCMA)
    cw.add_logger(cw2.cw_data.cw_pd_logger.PandasLogger())
    cw.run()
