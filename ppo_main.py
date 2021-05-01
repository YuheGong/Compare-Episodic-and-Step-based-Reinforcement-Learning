import os

import cma
import dill
import numpy as np
from cma.bbobbenchmarks import nfreefunclasses

import cw2.cluster_work
import cw2.experiment
import ppo_logger
from ppo_class import PPO_Holereacher

if __name__ == "__main__":
    cw = cw2.cluster_work.ClusterWork(PPO_Holereacher)
    cw_log = cw.add_logger(ppo_logger.PPOLogger())
    cw.run()




