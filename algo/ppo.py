import os
import sys
sys.path.append('..')
sys.path.append('..')
from utils.env import env_maker, env_save
from utils.logger import logging
import utils.callback as callback
from utils.callback import VecNormalizeCallback, DummyCallback
from utils.yaml import write_yaml, read_yaml
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.ppo import MlpPolicy
#from utils.bic_policy import CustomActorCriticPolicy, CustomGaussianDistribution



if __name__ == "__main__":

    # read config file

    file_name = "config/DeepMindBallInCupDense-v0.yml"
    data = read_yaml(file_name)

    # create log folder
    path = logging(data['env_params']['env_name'], data['algorithm'])
    data['path'] = path

    # choose the algorithm according to the config file
    ALGOS = {
        'a2c': A2C,
        'dqn': DQN,
        'ddpg': DDPG,
        'her': HER,
        'sac': SAC,
        'ppo': PPO,
        'td3': TD3
    }
    ALGO = ALGOS[data['algorithm']]

    # choose the tensorboard callback function according to the environment wrapper
    data['callback_class'] = callback.callback_function(data)
    CALLBACKS = {
        'VecNormalizeCallback': VecNormalizeCallback,
        'DummyCallback': DummyCallback
    }
    CALLBACK = CALLBACKS[data['callback_class']]


    # make the environment
    env = env_maker(data, num_envs=data["env_params"]['num_envs'])
    test_env = env_maker(data, num_envs=1, training=False, norm_reward=False)

    # make the model and save the model
    #CustomPolicy = CustomActorCriticPolicy
    model = ALGO(MlpPolicy, env, verbose=1, create_eval_env=True,
    #model = ALGO(MlpPolicy, env, verbose=1, create_eval_env=True,
                tensorboard_log=data['path'],
                seed=3,
                learning_rate=data["algo_params"]['learning_rate'],
                batch_size=data["algo_params"]['batch_size'],
                n_steps=data["algo_params"]['n_steps'])
    try:
        test_env_path = data['path'] + "/eval/"
        #print("test_env_path",test_env_path)
        model.learn(total_timesteps=int(data['algo_params']['total_timesteps']), eval_freq = 2048, n_eval_episodes = 8,
                    eval_log_path=test_env_path, eval_env=test_env)
    except KeyboardInterrupt:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env)
        print('')
        print('training interrupt, save the model and config file to '+ data["path"])
    else:
        # save the model
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env)
        print('')
        print('training FINISH, save the model and config file to ' + data['path'])