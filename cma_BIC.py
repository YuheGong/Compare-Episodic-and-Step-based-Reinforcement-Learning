import cma
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.logger import logging

if __name__ == "__main__":

    #env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"  # simple version
    #env_name = f'dmc_ball_in_cup-catch_dmp-v0'
    env_name = f'dmc_ball_in_cup-catch_detpmp-v0'
    # env_name = "alr_envs:ALRBallInACupDMP-v0"
    algorithm = 'cma'
    n_cpu = 1
    dim = 15
    # dim = 35
    n_samples = 1

    env = gym.make(env_name)

    params = np.zeros(10)  # 2 action_dimension * 5 dof + 2 learn_goal
    #params = np.random.randn(1, dim)
    #params[0][-13] = 2 * np.pi
    #params[0][-14] = - 2 * np.pi / 3
    #params[0][-15] = - np.pi

    algo = cma.CMAEvolutionStrategy(x0=params, sigma0=0.1, inopts={"popsize": 10})

    # logging
    env_name = "alr_envs:f'dmc_ball_in_cup-catch_dmp-v0'"
    path = logging(env_name, algorithm)
    log_writer = SummaryWriter(path)
    env.reset()

    t = 0
    opt = -10
    opts = []
    fitness = []



    try:
        while t < 5000 and opt < -1:
            print("----------iter {} -----------".format(t))
            solutions = np.vstack(algo.ask())
            for i in range(len(solutions)):
                # print(i, solutions[i])
                _, reward, __, ___ = env.step(solutions[i])
                env.reset()
                print('reward', -reward)

                fitness.append(-reward)
            # fitness = fitness[0].tolist()
            algo.tell(solutions, fitness)
            _, opt, __, ___ = env.step(algo.mean)
            env.reset()
            print("opt", -opt)

            np.save(path + "/algo_mean.npy", algo.mean)
            log_writer.add_scalar("iteration/reward", opt, t + 1)
            log_writer.add_scalar("iteration/dist_entrance", env.dist_entrance, t + 1)
            log_writer.add_scalar("iteration/dist_bottom", env.dist_bottom, t + 1)

            fitness = []
            t += 1

    except KeyboardInterrupt:
        np.save(path + "/algo_mean.npy", algo.mean)
        print('')
        print('training interrupt, save the model to ' + path)
    else:
        np.save(path + "/algo_mean.npy", algo.mean)
        print('')
        print('training Finish, save the model to ' + path)


