import argparse
import gym
import os
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG

import matplotlib.pyplot as plt



def make_env(env_id, rank):
    def _init():
        env = gym.make("alr_envs:" + env_id)
        return env
    return _init


def step_based(algo: str, env_id: str, model_id: str, step: str):
    path = "./logs/" + algo + "/" + env_id + "_" + model_id
    n_cpu = 1

    stats_file = algo.upper() + '.pkl'
    stats_path = os.path.join(path, stats_file)
    env = DummyVecEnv(env_fns=[make_env(env_id, i) for i in range(n_cpu)])
    env = VecNormalize.load(stats_path, env)

    model_file = algo.upper() + '.zip'
    model_path = os.path.join(path, model_file)

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
    if "DeepMind" in env_id:
        for i in range(int(step)):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            vedio = env.render(mode="rgb_array")
            plt.imshow(vedio)
            plt.pause(0.01)
            plt.draw()
        env.close()

    else:
        for i in range(int(step)):
            time.sleep(0.01)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            env.render()
        env.close()

def episodic():
    pass


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

    STEP_BASED = ["ppo", "sac"]
    EPISODIC = ["cmaes"]
    if algo in STEP_BASED:
        step_based(algo, env_id, model_id, step)
    elif algo in EPISODIC:
        episodic()
    else:
        print("the algorithm (--algo) is false or not implemented")

