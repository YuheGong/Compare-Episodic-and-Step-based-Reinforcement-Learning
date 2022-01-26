import argparse
from utils.env import env_maker, env_save
from utils.logger import logging
from utils.model import model_building, model_learn, cmaes_model_training
from utils.yaml import write_yaml, read_yaml
import numpy as np
import gym
import cma
from torch.utils.tensorboard import SummaryWriter
from utils.csv import csv_save


def step_based(algo: str, env_id: str, seed=None):
    file_name = algo +".yml"
    data = read_yaml(file_name)[env_id]

    # create log folder
    path = logging(data['env_params']['env_name'], data['algorithm'])
    data['path'] = path
    data['seed'] = seed

    # make the environment
    env = env_maker(data, num_envs=data["env_params"]['num_envs'])
    eval_env = env_maker(data, num_envs=1, training=False, norm_reward=False)

    # make the model and save the model
    model = model_building(data, env, seed)

    # csv file path
    data["path_in"] = data["path"] + '/' + data['algorithm'].upper() + '_1'
    data["path_out"] = data["path"] + '/data.csv'

    try:
        eval_env_path = data['path'] + "/eval/"
        model_learn(data, model, eval_env, eval_env_path)
    except KeyboardInterrupt:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env, eval_env)
        csv_save(data)
        print('')
        print('training interrupt, save the model and config file to ' + data["path"])
    else:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env, eval_env)
        csv_save(data)
        print('')
        print('training FINISH, save the model and config file to ' + data['path'])

def episodic(algo, env_id, stop_cri, seed=None):
    file_name = algo + ".yml"
    data = read_yaml(file_name)[env_id]
    env_name = data["env_params"]["env_name"]
    env = gym.make(env_name[2:-1], seed=seed)

    params = data["algo_params"]['x_init'] * np.random.rand(data["algo_params"]["dimension"])
    ALGOS = {
        'cmaes': cma,
    }
    if data["algorithm"] == "cmaes":
        algorithm = ALGOS[data["algorithm"]].CMAEvolutionStrategy(x0=params, sigma0=data["algo_params"]["sigma0"], inopts={"popsize": data["algo_params"]["popsize"]})

    # logging
    path = "alr_envs:" + env_id
    path = logging(path, algo)
    log_writer = SummaryWriter(path)

    t = 0
    opts = []
    success = False
    success_mean = []
    success_full = []

    try:
        if stop_cri:
            while t < data["algo_params"]["iteration"] and not success:
                algorithm, env, success_full, success_mean, path, log_writer, opts, t = \
                    try_dmp_sac(algorithm, env, success_full, success_mean, path, log_writer, opts, t)
        else:
            while t < data["algo_params"]["iteration"]:
                algorithm, env, success_full, success_mean, path, log_writer, opts, t = \
                    try_dmp_sac(algorithm, env, success_full, success_mean, path, log_writer, opts, t)
    except KeyboardInterrupt:
        data["path_in"] = path
        data["path_out"] = path + '/data.csv'
        csv_save(data)
        np.save(path + "/algo_mean.npy", algorithm.mean)
        print('')
        print('training interrupt, save the model to ' + path)
    else:
        data["path_in"] = path
        data["path_out"] = path + '/data.csv'
        csv_save(data)
        np.save(path + "/algo_mean.npy", algorithm.mean)
        print('')
        print('training Finish, save the model to ' + path)


def try_dmp_sac(algorithm, env, success_full, success_mean, path, log_writer, opts, t):
    fitness = []
    print("----------iter {} -----------".format(t))
    solutions = np.vstack(algorithm.ask())
    import torch
    # torch.nn.init.xavier_uniform(env.dynamical_net.weight)
    for i in range(len(solutions)):
        env.reset()
        _, reward, done, infos = env.step(solutions[i])
        success_full.append(env.env.success)
        print('reward', -reward)
        fitness.append(-reward)

        # env.optimizer.zero_grad()

    env.reset()

    '''
    import torch
    print("infos",infos["trajectory"].shape)
    print("actions", infos['step_actions'].shape)
    print("observations", infos['step_observations'].shape)
    loss = np.sum(infos["trajectory"] - infos['step_observations'],axis=1)
    print("shape", loss.shape)

    loss = torch.mean(torch.Tensor(loss))
    import tensorflow as tf
    #loss = tf.Variable(loss, requires_grad=True)
    loss_func = torch.nn.MSELoss()
    from torch.autograd import Variable
    #x = torch.unsqueeze(
    x = torch.unsqueeze(torch.Tensor(infos["trajectory"]), dim=1)
    y = torch.unsqueeze(torch.Tensor(infos['step_observations']), dim=1)
    x.requires_grad_()
    y.requires_grad_()
    from torch.autograd import Variable
    #x, y = (x, y)
    loss = loss_func(x, y)
    #loss.requres_grad = True
    #loss_func = torch.nn.MSELoss()
    #loss = loss_func(loss)
    loss.backward()
    env.optimizer.step()
    '''

    algorithm.tell(solutions, fitness)
    _, opt, __, ___ = env.step(algorithm.mean)

    np.save(path + "/algo_mean.npy", algorithm.mean)
    print("opt", -opt)
    opts.append(opt)
    t += 1

    success_mean.append(env.env.success)
    if success_mean[-1]:
        success_rate = 1
    else:
        success_rate = 0

    b = 0
    for i in range(len(success_full)):
        if success_full[i]:
            b += 1
    success_rate_full = b / len(success_full)
    success_full = []

    log_writer.add_scalar("iteration/success_rate_full", success_rate_full, t)
    log_writer.add_scalar("iteration/success_rate", success_rate, t)
    log_writer.add_scalar("iteration/reward", opt, t)
    log_writer.add_scalar("iteration/dist_entrance", env.env.dist_entrance, t)
    log_writer.add_scalar("iteration/dist_bottom", env.env.dist_bottom, t)
    # log_writer.add_scalar("iteration/dist_vec", env.env.dist_vec, t)
    for i in range(len(algorithm.mean)):
        log_writer.add_scalar(f"algorithm_params/mean[{i}]", algorithm.mean[i], t)
        log_writer.add_scalar(f"algorithm_params/covariance_matrix_mean[{i}]", np.mean(algorithm.C[i]), t)
        log_writer.add_scalar(f"algorithm_params/covariance_matrix_variance[{i}]", np.var(algorithm.C[i]), t)

    return algorithm, env, success_full, success_mean, path, log_writer, opts, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="the algorithm")
    parser.add_argument("--env_id", type=str, help="the environment")
    parser.add_argument("--stop_cri", type=str, help="whether you set up stop criterion or not")
    parser.add_argument("--seed", type=int, help="seed for training")

    args = parser.parse_args()

    if not args.algo and not args.env_id:
        parser.error('Please specify an algorithm (--algo) and an environment (--env_id) to train or enjoy')

    algo = args.algo
    env_id = args.env_id
    stop_cri = args.stop_cri
    STEP_BASED = ["ppo", "sac", "ddpg"]
    #print("algo", algo)
    EPISODIC = ["dmp", "promp"]
    if algo in STEP_BASED:
        step_based(algo, env_id, seed=args.seed)
    elif algo in EPISODIC:
        episodic(algo, env_id, stop_cri, seed=args.seed)
    else:
        print("the algorithm " + algo + " is false or not implemented")

