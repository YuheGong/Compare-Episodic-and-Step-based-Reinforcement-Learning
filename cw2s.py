import os
import alr_envs
import cma
import dill
import numpy as np
from cma.bbobbenchmarks import nfreefunclasses
import gym


import cw2.cluster_work
import cw2.cw_data.cw_pd_logger
import cw2.experiment
import os
import random

from cw2 import cluster_work, cw_error, experiment
from cw2.cw_data import cw_logging, cw_pd_logger


class CWCMA(cw2.experiment.AbstractIterativeExperiment):
    def __init__(self):
        super().__init__()
        self.env = None
        self.algorithm = None

    def initialize(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:

        dim = config.params.optim_params.dim
        x_start = config.params.optim_params.x_init * np.random.randn(dim)
        init_sigma = config.params.optim_params.init_sigma
        self.env = gym.make(config.params.env_name[2:-1])

        self.algorithm  = cma.CMAEvolutionStrategy(
            x0=x_start,
            sigma0=init_sigma,
            inopts={
                'popsize': config.params.optim_params.n_samples
            }
        )

        self.env.reset()



    def iterate(self, config: dict, rep: int, n: int) -> dict:
        opt = -10
        opts = []
        opt_full = []
        fitness = []
        success = False
        success_mean = []
        success_full = []
        success_rate = 0
        success_rate_full = 0


        
        print("----------iter {} -----------".format(n))
        solutions = np.vstack(self.algorithm.ask())
        for i in range(len(solutions)):
            # print(i, solutions[i])
            _, reward, __, ___ = self.env.step(solutions[i])
            success_full.append(self.env.success)
            self.env.reset()
            print('reward', -reward)
            opt_full.append(reward)
            fitness.append(-reward)

        self.algorithm.tell(solutions, fitness)
        _, opt, __, ___ = self.env.step(self.algorithm.mean)

        # print("success", env.env.success)
        # assert 1==9
        success = True
        # print("success", success)
        success_mean.append(self.env.env.success)
        self.env.reset()
        print("opt", -opt)

        #np.save(path + "/algo_mean.npy", self.algorithm.mean)
        #log_writer.add_scalar("iteration/reward", opt, (t + 1) * 10 * 250)
        #log_writer.add_scalar("iteration/dist_entrance", env.env.dist_entrance, (t + 1) * 10 * 250)
        #log_writer.add_scalar("iteration/dist_bottom", env.env.dist_bottom, (t + 1) * 10 * 250)

        fitness = []
        opts.append(opt)
        # opt_full.append(reward)
        n += 1

        if n % 10 == 0:
            a = 0
            b = 0

            # print(len(opts))
            for i in range(len(success_mean)):

                if success_mean[i]:
                    a += 1
            success_rate = a / len(success_mean)
            # print(a)
            success_mean = []
            #log_writer.add_scalar("iteration/success_rate", success_rate, (n + 1) * 10 * 250)

            for i in range(len(success_full)):
                if success_full[i]:
                    b += 1
            success_rate_full = b / len(success_full)
            #success_full = []
            # print("success_full_rate", success_rate_full)

        
        # do one iteration of cma es
        

        results_dict = {"reward": opt,
                        "dist_entrance": self.env.env.dist_entrance,
                        "dist_bottom": self.env.env.dist_bottom,
                        "success_rate": success_rate,
                        "success_rate_full": success_rate_full,
                        "total_samples": (n + 1) * config.params.optim_params.n_samples
                        }

        return results_dict

    def save_state(self, config: dict, rep: int, n: int) -> None:
        #if n % 50 == 0:
        #    f_name = os.path.join(
        #        config.rep_log_paths[rep], 'optimizer.pkl')
        #    with open(f_name, 'wb') as f:
        #        dill.dump(self.optimizer, f)
        pass

    def finalize(self, surrender = None, crash: bool = False):
        pass

    def restore_state(self):
        pass


if __name__ == "__main__":
    cw = cw2.cluster_work.ClusterWork(CWCMA)
    cw.add_logger(cw2.cw_data.cw_pd_logger.PandasLogger())
    cw.run()
