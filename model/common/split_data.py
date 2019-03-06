import os
from utils.envs import *
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 42

if __name__ == '__main__':
    os.rename(beauty_val_repo, beauty_test_repo)
    print('Renaming beauty_data_info_val_competition.csv to beauty_data_info_test_competition.csv')
    os.rename(fashion_val_repo, fashion_test_repo)
    print('Renaming fashion_data_info_val_competition.csv to fashion_data_info_test_competition.csv')
    os.rename(mobile_val_repo, mobile_test_repo)
    print('Renaming mobile_data_info_val_competition.csv to mobile_data_info_test_competition.csv')

    beauty_train = pd.read_csv(beauty_train_repo)
    beauty_dev, beauty_val = train_test_split(beauty_train, test_size=0.20, random_state=SEED)
    beauty_dev.to_csv(beauty_dev_repo, index=False)
    beauty_val.to_csv(beauty_val_repo, index=False)
    print('Splitting training dataset into beauty_data_info_dev_competition.csv and'
          ' beauty_data_info_val_competition.csv')

    fashion_train = pd.read_csv(fashion_train_repo)
    fashion_dev, fashion_val = train_test_split(fashion_train, test_size=0.20, random_state=SEED)
    fashion_dev.to_csv(fashion_dev_repo, index=False)
    fashion_val.to_csv(fashion_val_repo, index=False)
    print('Splitting training dataset into fashion_data_info_dev_competition.csv and'
          ' fashion_data_info_val_competition.csv')

    mobile_train = pd.read_csv(mobile_train_repo)
    mobile_dev, mobile_val = train_test_split(mobile_train, test_size=0.20, random_state=SEED)
    mobile_dev.to_csv(mobile_dev_repo, index=False)
    mobile_val.to_csv(mobile_val_repo, index=False)
    print('Splitting training dataset into mobile_data_info_dev_competition.csv and'
          ' mobile_data_info_val_competition.csv')

