import os
import cma
import gym
import tempfile
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from alr_envs.utils.mp_env_async_sampler import AlrMpEnvSampler
from logger import logging
#from alr_envs.mujoco.ball_in_a_cup import ball_in_a_cup

if __name__ == "__main__":

    env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"  # simple version
    #env_name = "alr_envs:ALRBallInACupDMP-v0"
    algorithm = 'cma'
    n_cpu = 1
    dim = 15
    #dim = 35
    n_samples = 1

    sampler = AlrMpEnvSampler(env_name, num_envs=n_cpu, seed = 5000)

    thetas = np.random.randn(n_samples, dim)
    fitness = []
    params =  np.zeros((1,dim))
    # params = np.random.randn(1, dim)

    params[0][-13] = 2 * np.pi
    params[0][-14] = - 2 * np.pi / 3
    params[0][-15] = - np.pi
    #print('param', params)
    algo = cma.CMAEvolutionStrategy(x0=params, sigma0=0.1, inopts={"popsize": 14})

    # logging
    path = logging(env_name, algorithm)
    log_writer = SummaryWriter(path)

    t = 0
    opt = 0
    opts = []
    while t < 500 and opt < 0.92:
        print("----------iter {} -----------".format(t))
        solutions = np.vstack(algo.ask())
        _, reward, __, ___ = sampler(solutions)
        print('reward', -reward)

        fitness.append(-reward)
        fitness = fitness[0].tolist()
        algo.tell(solutions, fitness)
        _, opt, __, ___ = sampler(algo.mean)
        print("opt", -opt)
       
        np.save(path + "/algo_mean.npy", algo.mean)
        log_writer.add_scalar("iteration/reward", opt, t+1)
       
        fitness = []
        t += 1

    np.save(path + "/algo_mean.npy", algo.mean)
