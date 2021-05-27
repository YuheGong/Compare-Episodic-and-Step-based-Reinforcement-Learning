import gym
import os
from gym import wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        # env = wrappers.Monitor(env, path, force=True)
        return env

    return _init

def env_maker(data: dict):
    if data["env_params"]['wrapper'] == "VecNormalize":
        env = DummyVecEnv(env_fns=[make_env(data["env_params"]['env_name'], data['path'], i) for i in range(data["env_params"]['num_envs'])])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    else:
        env = gym.make(data["env_params"]['env_name'])
    return env

def env_save(data: dict, model, env):
    model_path = os.path.join(data['path'], "PPO.zip")
    model.save(model_path)
    if 'VecNormalize' in data['env_params']['wrapper']:
        stats_path = os.path.join(data['path'], "PPO.pkl")
        env.save(stats_path)


def env_continue_load(data: dict):
    if data["env_params"]['wrapper'] == "VecNormalize":
        env = DummyVecEnv(env_fns=[make_env(data["env_params"]['env_name'], data['path'], i) for i in range(data["env_params"]['num_envs'])])
        stats_path = os.path.join(data['continue_path'], data['algorithm'].upper() + '.pkl')
        env = VecNormalize.load(stats_path, env)
    else:
        env = gym.make(data["env_params"]['env_name'])
    return env