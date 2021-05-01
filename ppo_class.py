import os
import datetime as dt
import abc
import cma
import dill
import numpy as np
from cma.bbobbenchmarks import nfreefunclasses
from cw2.cw_data import cw_logging
from cw2.cw_error import ExperimentSurrender

import cw2.cluster_work
import cw2.cw_data.cw_pd_logger
import PPOexperiment
from stable_baselines3 import PPO
import numpy as np
from alr_envs.classic_control.dense_hole_reacher import DenseHoleReacher
import torch as th

from gym import wrappers
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import pickle
import tensorflow as tf
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


class PPO_Holereacher(PPOexperiment.AbstractIterativeExperiment):
    def __init__(self):
        super().__init__()
        self.env = None
        self.model = None

    def initialize(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:
        self.config = config
        self.env_initialize()
        self.model_initialize()


    def env_initialize(self):#, config: dict, logger: cw_logging.AbstractLogger) -> None:
        def make_env(rank, seed=0):
            def _init():
                env = DenseHoleReacher(num_links=5,
                                  allow_self_collision=False,
                                  allow_wall_collision=False,
                                  hole_width=0.15,
                                  hole_depth=1,
                                  hole_x=1,
                                  collision_penalty=10000)

                env = wrappers.Monitor(env, 'tmp', force=True)
                return env
            return _init
        n_envs = 8
        self.env = DummyVecEnv(env_fns=[make_env(i) for i in range(n_envs)])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10.)

    def model_initialize(self):#, config: dict):#, nv)# logger: cw_logging.AbstractLogger) -> None:

        #if model_name == 'PPO':
        self.model = PPO(MlpPolicy, self.env, verbose=1,
                    # policy_kwargs=policy_kwargs,
                    #tensorboard_log="./ppo_1_wid/",
                    learning_rate=0.0001,
                    n_steps=2048)  ######200)
        #else:
        #    from stable_baselines3 import model_name
        #    self.model = model_name("MlpPolicy", env, verbose=1,
        #            # policy_kwargs=policy_kwargs,
        #            tensorboard_log="./ppo_1_wid/",
        #            learning_rate=0.0001,
        #            n_steps=2048)  ######200)


    def iterate(self, config: dict, rep: int, n: int) -> dict:#, config: dict) -> dict:
        #self.env_initialize()
        #self.model_initialize()
        self.model.learn(total_timesteps=int(4e4))
        self.env.close()
        return {}

    def save_state(self, rep_path: dict) -> None:#, config: dict, rep: int, n: int) -> None:
        '''
        model_save = os.path.join(self.config.path, 'model.zip')
        env_save = (self.config.path, 'env.pkl')
        '''
        model_path = rep_path[0]
        env_path = rep_path[1]

        self.model.save(model_path)
        self.env.save(env_path)

    def render(self):
        stats_path = os.path.join(self.config.path, "env.pkl")
        model_path = os.path.join(self.config.path, "model.zip")
        env = VecNormalize.load(stats_path, env)
        model = PPO.load(model_path)
        #obs = env.reset()
        #for i in range(200):
            ##action, _states = model.predict(obs, deterministic=True)
            #obs, rewards, dones, info = env.step(action)
            #env.render()
        #env.close()


    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        pass

    def restore_state(self):
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:

        for n in range(config["iterations"]):
            surrender = False
            try:
                res = self.iterate(config, rep, n)
            except ExperimentSurrender as e:
                res = e.payload
                surrender = True

            res["ts"] = dt.datetime.now()
            #res["rep"] = rep
            #res["iter"] = n

            logger.process([self.model, self.env])

            if surrender:
                raise ExperimentSurrender()

        logger.finalize()
