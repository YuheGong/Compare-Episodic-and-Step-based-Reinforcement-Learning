import argparse
from utils.model import model_learn
import os
from utils.env import env_maker, env_save, env_continue_load
from utils.logger import logging
from utils.yaml import write_yaml, read_yaml
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG


def step_based(algo: str, env_id: str, model_id: str):
    file_name =  algo +".yml"
    data = read_yaml(file_name)[env_id]

    # create log folder
    path = logging(data['env_params']['env_name'], data['algorithm'])
    data['path'] = path
    data["continue"] = True
    data['continue_path'] = "logs/ppo/" + env_id + "_" + model_id

    # choose the algorithm according to the algo
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


    # make the environment
    env = env_continue_load(data)
    test_env = env_maker(data, num_envs=1, training=False, norm_reward=False)

    # make the model and save the model
    model_path = os.path.join(data['continue_path'], data['algorithm'].upper() + '.zip')
    model = ALGO.load(model_path, tensorboard_log=data['path'])
    model.set_env(env)

    try:
        test_env_path = data['path'] + "/eval/"
        model_learn(data, model, test_env, test_env_path)
    except KeyboardInterrupt:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env)
        print('')
        print('continune-training interrupt, save the model and config file to ' + data["path"])
    else:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env)
        print('')
        print('continue-training FINISH, save the model and config file to ' + data['path'])

def episodic():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="the algorithm")
    parser.add_argument("--env_id", type=str, help="the environment")
    parser.add_argument("--model_id", type=str, help="the model")
    args = parser.parse_args()

    if not args.algo and not args.env_id:
        parser.error('Please specify an algorithm (--algo) and an environment (--env_id) to train or enjoy')

    algo = args.algo
    env_id = args.env_id
    model_id = args.model_id

    STEP_BASED = ["ppo", "sac"]
    EPISODIC = ["cmaes"]
    if algo in STEP_BASED:
        step_based(algo, env_id, model_id)
    elif algo in EPISODIC:
        episodic()
    else:
        print("the algorithm (--algo) is false or not implemented")


