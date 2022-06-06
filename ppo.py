import gym
from utils.logger import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Parallel environments
env_id = "alr_envs:ALRReacherBalanceIP-v3"
path=logging(env_id, "ppo")
env = make_vec_env(env_id, n_envs=8)
test_env = make_vec_env(env_id, n_envs=1)
net_arch = {}
net_arch["pi"] = [int(256), int(256)]
net_arch["vf"] = [int(256), int(256)]
net_arch = [dict(net_arch)]


model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=net_arch),
                     tensorboard_log=path, #create_eval_env=True,
                     seed=0,
                     learning_rate=0.0001,
                     batch_size=200,
                     n_steps=2000)

eval_callback = EvalCallback(test_env, best_model_save_path=path,  n_eval_episodes=1,
                                 log_path=path, eval_freq=225 * 8,
                                 deterministic=False, render=False)

model.learn(total_timesteps=2.e6)