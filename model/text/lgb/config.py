import os
import pandas as pd

class Config:
    def __init__(self):
        self.n_svd = int(self.retrieve('N_SVD'))
        self.n_round_eta = int(self.retrieve('N_ROUND_ETA'))
        self.n_round = int(self.retrieve('N_ROUND'))
        self.n_threads = int(self.retrieve('N_THREADS'))

    def retrieve(self, env_name):
        value = os.getenv(env_name)
        assert value is not None, 'environment variable {} is not set'.format(env_name)
        return value

    def save(self, path):
        df = pd.DataFrame([{
            'n_svd': self.n_svd,
            'n_round_eta': self.n_round_eta,
            'n_round': self.n_round
        }])
        df.to_csv(path, index=False)

config = Config()
