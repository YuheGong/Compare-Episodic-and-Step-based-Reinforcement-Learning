from cw2.cw_data import cw_logging
import cw2.cluster_work
import cw2.cw_data.cw_pd_logger
import cw2.experiment
from algorithm.cma.cma_class import CMAHolereacher

if __name__ == "__main__":

    cw = cw2.cluster_work.ClusterWork(CMAHolereacher)
    cw.add_logger(cw2.cw_data.cw_pd_logger.PandasLogger())
    cw.run()

