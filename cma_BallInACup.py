import os
import cma
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from alr_envs.utils.mp_env_async_sampler import AlrMpEnvSampler
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
    #params = 3*np.random.randn(1, dim)
    params =  np.zeros((1,dim))

    params[0][-13] = 8* np.pi/4
    params[0][-14] = -2 * np.pi/3
    params[0][-15] = -np.pi
    print('param', params)
    algo = cma.CMAEvolutionStrategy(x0=params, sigma0=0.1, inopts={"popsize": 14})

    # create log folder
    env_log_index = env_name.index(':')
    env_log_name = env_name[env_log_index+1:]
    path = "logs/" + algorithm + '/'
    folders = os.listdir(path)
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
    print('log into: ' + path)

    t = 0
    opt = 0
    opts = []
    while t < 500 and opt <0.92:
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
