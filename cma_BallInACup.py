import cma
import gym
import numpy as np

from alr_envs.utils.mp_env_async_sampler import AlrMpEnvSampler
from alr_envs.mujoco.ball_in_a_cup import ball_in_a_cup
#from alr_envs.utils.legacy.dmp_async_vec_env import DmpAsyncVectorEnv, _worker

if __name__ == "__main__":

    #a = ball_in_a_cup.ALRBallInACupEnv(reward_type = "no_context")
    #a.render()

    env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"
    n_cpu = 8
    dim = 15
    n_samples = 1

    sampler = AlrMpEnvSampler(env_name, num_envs=n_cpu)

    thetas = np.random.randn(n_samples, dim)  # usually form a search distribution
    #_, rewards, __, ___ = sampler(thetas)
    fitness = []
    params = np.random.randn(1, dim)
    algo = cma.CMAEvolutionStrategy(x0=params, sigma0=0.1, inopts={"popsize": 14})
    #print(solutions.shape)
    t = 0
    opt = 1e10
    opts = []
    while t < 2 and opt > 1e-8:
        print("----------iter {} -----------".format(t))

        solutions = np.vstack(algo.ask())
        #solutions = np.vstack(algo.ask())
        #print("solution.shape[0]",solutions.shape[0])
        for i in range(solutions.shape[0]):
            _, reward, __, ___ = sampler(solutions)
            #reward = - reward
            fitness.append(reward)
        #print(fitness)
        fitness = fitness[0].tolist()

        algo.tell(solutions, fitness)
        _, opt, __, ___ = sampler(algo.mean)
        print(opt)
        fitness = []
        t += 1

    env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"
    test_env = gym.make(env_name)
    test_env.render("human")
    test_env.step(algo.mean)


    # test_env = make_env()
    #test_env.rollout(algo.mean, render=True)

    """
    params = np.random.randn(15)
    #env.step(params)
    hyperparams = {'n_samples': 14, 'context': 'spawn', 'shared_memory': False,
                   'worker': _worker}
    env = make_env()
    algo = cma.CMAEvolutionStrategy(x0=params, sigma0=0.1, inopts={"popsize": 14})

    t = 0
    opt = 1e10
    opts = []
    while t < 100 and opt > 1e-8:
        print("----------iter {} -----------".format(t))

        # sample parameters to test
        solutions = algo.ask()
        print('solutions', solutions)
        # collect rollouts with parameters, need to negate because cma-es minimizes
        fitness = -env(np.vstack(solutions))[0]
        # update search distributioon
        algo.tell(solutions, fitness)
        opt = -env(algo.mean)[0][0]
        opts.append(opt)
        print(opt)

        t += 1
        """

