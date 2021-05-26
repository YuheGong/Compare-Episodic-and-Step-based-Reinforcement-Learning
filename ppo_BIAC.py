import gym
import yaml
from gym import wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.ppo import MlpPolicy
import os
from utils.logger import logging
import utils.callback as callback
from utils.config import write_yaml


if __name__ == "__main__":

    algorithm = 'ppo'
    timesteps = 5e6
    learning_rate = 0.0001
    batch_size = 50
    n_steps = 2000


    env_name = 'alr_envs:ALRBallInACupSimpleDense-v0'
    path = logging(env_name, algorithm)

    n_envs = 8

    data = {
        "algo": algorithm,
        "env_name" : env_name,
        "n_envs" : 8,
        "path" : path,
        "algo": {
            "timesteps": timesteps,
            "learning_rate": learning_rate,
            "batch_size" : batch_size,
            "n_steps" : n_steps
        },
    }



    def make_env(env_name, path, rank, seed=0):

        def _init():
            env = gym.make(env_name)
            #env = wrappers.Monitor(env, path, force=True)
            return env

        return _init

    env = DummyVecEnv(env_fns=[make_env(env_name, path, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    #env = gym.make(env_name)

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
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_steps=n_steps)


    try:
        model.learn(total_timesteps=int(timesteps), callback=callback.VevNormalizeCallback())  # , callback=TensorboardCallback())
    except KeyboardInterrupt:
        data['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        model_path = os.path.join(path, "PPO.zip")
        model.save(model_path)
        print('')
        print('training interrupt, save the model and config file')
    else:
        # save the model
        data['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        model_path = os.path.join(path, "PPO.zip")
        model.save(model_path)
        print('training finish, save the model and config file')
        #stats_path = os.path.join(path, "PPO.pkl")
        #env.save(stats_path)
