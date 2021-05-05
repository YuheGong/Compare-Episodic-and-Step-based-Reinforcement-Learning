import os
import cma
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from alr_envs.utils.mp_env_async_sampler import AlrMpEnvSampler
#from alr_envs.mujoco.ball_in_a_cup import ball_in_a_cup

if __name__ == "__main__":

    env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"
    n_cpu = 8
    dim = 15
    n_samples = 1

    sampler = AlrMpEnvSampler(env_name, num_envs=n_cpu)

    thetas = np.random.randn(n_samples, dim)
    fitness = []
    params = np.random.randn(1, dim)
    algo = cma.CMAEvolutionStrategy(x0=params, sigma0=0.1, inopts={"popsize": 14})

    #write tensorboard
    env_log_index = env_name.index(':')
    env_log_name = env_name[env_log_index+1:]
    path = "logs/"
    folders = os.listdir(path)
    print(folders)
    if folders == []:
        path = path + env_log_name + "_1"
    else:
        s = 0
        for folder in folders:
            if int(folder[-1]) > s:
                s = int(folder[-1])
        s += 1
        path = path + env_log_name + '_' + str(s)
    log_writer = SummaryWriter(path)
    print('log into' + path + ": ")

    t = 0
    opt = 1e10
    opts = []
    while t < 500 and opt > 1e-8:
        print("----------iter {} -----------".format(t))
        solutions = np.vstack(algo.ask())
        #for i in range(solutions.shape[0]):
        _, reward, __, ___ = sampler(solutions)
        print('reward', reward)

        fitness.append(reward)
        fitness = fitness[0].tolist()
        algo.tell(solutions, fitness)
        _, opt, __, ___ = sampler(algo.mean)
        print("opt", opt)
       
        np.save(path + "/algo_mean.npy", algo.mean)
        log_writer.add_scalar("iteration/reward", opt, t+1)
       
        fitness = []
        t += 1

    np.save(path + "/algo_mean.npy", algo.mean)
