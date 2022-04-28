import argparse
import gym
import os
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
import numpy as np
import matplotlib.pyplot as plt
from utils.yaml import write_yaml, read_yaml
import alr_envs
from stable_baselines3.common.vec_env.obs_dict_wrapper import  ObsDictWrapper


def make_env(env_id, rank):
    def _init():
        env = gym.make("alr_envs:" + env_id)
        return env
    return _init


def step_based(algo: str, env_id: str, model_id: str, step: str):
    path = "./logs/" + algo + "/" + env_id + "_" + model_id
    num_envs = 1
    stats_file = 'env_normalize.pkl'
    stats_path = os.path.join(path, stats_file)
    env = DummyVecEnv(env_fns=[make_env(env_id, i) for i in range(num_envs)])
    #env = VecNormalize.load(stats_path, env)
    #env = ObsDictWrapper(env)
    env = gym.make("alr_envs:" + env_id)

    model_path = os.path.join(path, "eval/best_model.zip")

    # model_path = os.path.join(path, "model.zip")

    ALGOS = {
        'a2c': A2C,
        'dqn': DQN,
        'ddpg': DDPG,
        'her': HER,
        'sac': SAC,
        'ppo': PPO,
        'td3': TD3
    }
    ALGO = ALGOS[algo]
    model = ALGO.load(model_path)

    obs = env.reset()
    rewards = 0
    if "DeepMind" in env_id:
        for i in range(int(step)):
            #time.sleep(0.1)
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, dones, info = env.step(action)
            rewards += reward
            #env.render(mode="rgb_array")
            env.render(mode="human")
        print("rewards", rewards)
        env.close()
    elif "Meta" in env_id:
        print("meta")
        for i in range(int(step)):
            time.sleep(0.01)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            env.render(False)
        #env.close()

    else:
        for i in range(int(step)):
            #time.sleep(0.1)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            env.render()
        env.close()

def episodic(algo: str, env_id, model_id: str, step: str, seed=None):
    file_name = algo + ".yml"
    data = read_yaml(file_name)[env_id]
    env_name = data["env_params"]["env_name"]

    path = "logs/" + algo + "/" + env_id + "_" + model_id + "/algo_mean.npy"
    algorithm = np.load(path)
    print("algorithm", algorithm)

    if 'Meta' in env_id:
        from alr_envs.utils.make_env_helpers import make_env
        env = make_env(env_name, seed=seed)
    else:
        env = gym.make(env_name[2:-1])
    env.reset()

    if "DeepMind" in env_id:
        env.render("rgb_array")
        env.step(algorithm)
    elif "Meta" in env_id:
        env.render(mode="meta")
        env.step(algorithm)
    else:
        env.render()
        env.step(algorithm)
    env.render()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="the algorithm")
    parser.add_argument("--env_id", type=str, help="the environment")
    parser.add_argument("--model_id", type=str, help="the model")
    parser.add_argument("--step", type=str, help="how many steps rendering")

    args = parser.parse_args()

    if not args.algo and not args.env_id:
        parser.error('Please specify an algorithm (--algo) and an environment (--env_id) to train or enjoy')

    algo = args.algo
    env_id = args.env_id
    model_id = args.model_id
    step = args.step

    STEP_BASED = ["ppo", "sac", "td3"]
    EPISODIC = ["dmp", "promp"]



    if algo in STEP_BASED:
        step_based(algo, env_id, model_id, step)
    elif algo in EPISODIC:
        episodic(algo, env_id, model_id, step)
    else:
        print("the algorithm (--algo) is false or not implemented")

