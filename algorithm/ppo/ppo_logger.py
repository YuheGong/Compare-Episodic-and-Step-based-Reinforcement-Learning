import os

import attrdict
import pandas as pd
from cw2.cw_data import cw_logging

class PPOLogger(cw_logging.AbstractLogger):
    """Writes the results of each repetition seperately to disk
    Each repetition is saved in its own directory. Write occurs after every iteration.
    """

    def __init__(self, ignore_keys: list = []):
        self.log_path = ""
        self.csv_name = "rep.csv"
        self.pkl_name = "rep.pkl"
        self.df = pd.DataFrame()
        self.ignore_keys = ignore_keys
        #self.index = 0

    def initialize(self, config: attrdict.AttrDict, rep: int, rep_log_path: str):
        self.log_path = rep_log_path
        self.csv_name = os.path.join(self.log_path, 'rep_{}.csv'.format(rep))
        self.pkl_name = os.path.join(self.log_path, 'rep_{}.pkl'.format(rep))
        self.df = pd.DataFrame()
        self.model_path = os.path.join(self.log_path, 'model')
        self.env_path = os.path.join(self.log_path, 'env.pkl')

    def process(self, data: dict):

        model = data[0]
        env = data[1]
        try:
            model.save(self.model_path)
        except:
            cw_logging.getLogger().warning('Could not save model')

        try:
            env.save(self.env_path)
        except:
            cw_logging.getLogger().warning('Could not save env')



    def finalize(self) -> None:
        pass

    def load(self):
        payload = {}
        df: pd.DataFrame = None
        
        # Check if file exists
        try:
            df = pd.read_pickle(self.pkl_name)
        except FileNotFoundError as _:
            warn = "{} does not exist".format(self.pkl_name)
            cw_logging.getLogger().warning(warn)
            return warn

        # Enrich Payload with descriptive statistics for loading DF structure
        """
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                payload['{}_min'.format(c)] = df[c].min()
                payload['{}_max'.format(c)] = df[c].max()
                payload['{}_mean'.format(c)] = df[c].mean()
                payload['{}_std'.format(c)] = df[c].std()

            payload['{}_last'.format(c)] = df[c].iloc[-1]
        """
        payload[self.__class__.__name__] = df
        return payload
