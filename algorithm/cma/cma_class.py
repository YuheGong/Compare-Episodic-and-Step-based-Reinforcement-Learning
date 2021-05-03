import cma
import gym
import numpy as np

from alr_envs.utils.legacy.dmp_async_vec_env import DmpAsyncVectorEnv, _worker
from alr_envs.utils.legacy.dmp_env_wrapper import DmpEnvWrapper
from cw2.cw_data import cw_logging
from cw2.cw_error import ExperimentSurrender
import cw2.experiment

class CMAHolereacher(cw2.experiment.AbstractIterativeExperiment):
    def __init__(self):
        super().__init__()
        self.env = None
        self.optimizer = None
        self.ite = 1
        self.config = None

    def initialize(self, config: dict, rep: int, logger: cw_logging.AbstractLogger) -> None:
        self.config = config
        self.env = self.make_env()
        dim = self.config.params.optim_params.dim  # [2, 5, 10, 20]
        x_start = self.config.params.optim_params.x_init * np.random.randn(dim, 1)  # x_init: 1
        x_start[-5] = np.pi / 2
        x_start[-4] = -np.pi / 4
        x_start[-3] = -np.pi / 4
        x_start[-2] = -np.pi / 4
        x_start[-1] = -np.pi / 4
        init_sigma = self.config.params.optim_params.init_sigma
        popsize = self.config.params.optim_params.popsize
        self.optimizer = cma.CMAEvolutionStrategy(x0=x_start,
                                                  sigma0=init_sigma,
                                                  inopts={"popsize": popsize})

    def make_env(self):
        hyperparams = {'n_samples': self.config.env_params.asyn_env.hyperparams.n_samples,
                       'context': self.config.env_params.asyn_env.hyperparams.context,
                       'shared_memory': self.config.env_params.asyn_env.hyperparams.shared_memory,
                       'worker': self.config.env_params.asyn_env.hyperparams.worker}
        def maker(rank):
            def _init():
                env = gym.make(self.config.env_params.env_name)
                env = DmpEnvWrapper(env,
                                    num_dof=int(self.config.env_params.dmp_wrapper.num_dof),
                                    num_basis=int(self.config.env_params.dmp_wrapper.num_basis),
                                    duration=int(self.config.env_params.dmp_wrapper.duration),
                                    # dt=env._dt,
                                    learn_goal=self.config.env_params.dmp_wrapper.learn_goal,
                                    policy_type=self.config.env_params.dmp_wrapper.policy_type,
                                    alpha_phase=int(self.config.env_params.dmp_wrapper.alpha_phase))
                env.seed(self.config.base_seed + rank)
                return env
            return _init
        n_envs = self.config.env_params.asyn_env.n_envs
        return DmpAsyncVectorEnv(env_fns=[maker(i) for i in range(n_envs)], **hyperparams)

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        print("----------iter {} -----------".format(self.ite))
        solutions = self.optimizer.ask()
        fitness = -self.env(np.vstack(solutions))[0]
        self.optimizer.tell(solutions, fitness)
        opt = -self.env(self.optimizer.mean)[0][0]

        print(opt)

        self.ite += 1

    def save_state(self, config: dict, rep: int, n: int) -> None:
        self.env.close()
        test_env = self.maker(1, config)
        test_env.rollout(self.optimizer.mean, render=True)

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        # def finalize(self):
        pass

    def restore_state(self):
        pass
