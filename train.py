import argparse
from utils.env import env_maker, env_save
from utils.logger import logging
from utils.model import model_building, model_learn
from utils.yaml import write_yaml, read_yaml
import numpy as np
import gym
import cma
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG


def step_based(algo: str, env_id: str):
    file_name = algo +".yml"
    data = read_yaml(file_name)[env_id]

    # create log folder
    path = logging(data['env_params']['env_name'], data['algorithm'])
    data['path'] = path

    # make the environment
    env = env_maker(data, num_envs=data["env_params"]['num_envs'])
    test_env = env_maker(data, num_envs=1, training=False, norm_reward=False)

    # make the model and save the model
    model = model_building(data, env)

    try:
        test_env_path = data['path'] + "/eval/"
        model_learn(data, model, test_env, test_env_path)

    except KeyboardInterrupt:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env)
        print('')
        print('training interrupt, save the model and config file to ' + data["path"])
    else:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env)
        print('')
        print('training FINISH, save the model and config file to ' + data['path'])

def episodic(algo, env_id):
    file_name = algo + ".yml"
    data = read_yaml(file_name)[env_id]
    env_name = data["env_params"]["env_name"]
    print("env_name", env_name)
    env = gym.make(env_name[2:-1])

    params = np.zeros(data["algo_params"]["dimension"])
    ALGOS = {
        'cmaes': cma,
    }
    if algo == "cmaes":
        algorithm = ALGOS[algo] .CMAEvolutionStrategy(x0=params, sigma0=data["algo_params"]["sigma0"], inopts={"popsize": data["algo_params"]["popsize"]})

    # logging
    path = "alr_envs:" + env_id
    path = logging(path, algo)
    log_writer = SummaryWriter(path)
    env.reset()

    t = 0
    opt = -10
    fitness = []

    try:
        while t < 1000:  # and opt < 3.2 :# and opt < -1:
            print("----------iter {} -----------".format(t))
            solutions = np.vstack(algorithm.ask())
            for i in range(len(solutions)):
                # print(i, solutions[i])
                _, reward, __, ___ = env.step(solutions[i])
                env.reset()
                print('reward', -reward)
                fitness.append(-reward)

            algorithm.tell(solutions, fitness)
            _, opt, __, ___ = env.step(algorithm.mean)
            env.reset()
            print("opt", -opt)

            np.save(path + "/algo_mean.npy", algorithm.mean)
            log_writer.add_scalar("iteration/reward", opt, t + 1)
            log_writer.add_scalar("iteration/dist_entrance", env.env.dist_entrance, t + 1)
            log_writer.add_scalar("iteration/dist_bottom", env.env.dist_bottom, t + 1)

            fitness = []
            t += 1

    except KeyboardInterrupt:
        np.save(path + "/algo_mean.npy", algorithm.mean)
        print('')
        print('training interrupt, save the model to ' + path)
    else:
        np.save(path + "/algo_mean.npy", algorithm.mean)
        print('')
        print('training Finish, save the model to ' + path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="the algorithm")
    parser.add_argument("--env_id", type=str, help="the environment")
    args = parser.parse_args()

    if not args.algo and not args.env_id:
        parser.error('Please specify an algorithm (--algo) and an environment (--env_id) to train or enjoy')

    algo = args.algo
    env_id = args.env_id

    STEP_BASED = ["ppo", "sac"]
    EPISODIC = ["cmaes"]
    if algo in STEP_BASED:
        step_based(algo, env_id)
    elif algo in EPISODIC:
        episodic(algo, env_id)
    else:
        print("the algorithm (--algo) is false or not implemented")

