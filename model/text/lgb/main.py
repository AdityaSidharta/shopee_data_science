import argparse
import pandas as pd
from sklearn.externals import joblib

from model.text.lgb.config import config
from model.text.lgb.lgb_model import create_label, create_prediction
from utils.common import create_directory
from utils.common import get_datetime
from utils.envs import *
from utils.logger import logger

beauty_topic = ['Colour_group', 'Brand', 'Benefits', 'Product_texture', 'Skin_type']
fashion_topic =['Pattern', 'Collar Type', 'Sleeves', 'Fashion Trend',
                'Clothing Material']
mobile_topic = ['Operating System', 'Features',
               'Network Connections', 'Memory RAM', 'Brand', 'Warranty Period',
               'Storage Capacity', 'Color Family', 'Phone Model', 'Camera',
               'Phone Screen Size']


if __name__ == '__main__':
    logger.setup_logger('lgb')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", help='Using Development Dataset and Validation Dataset instead to perform training',
                        action='store_true')
    args = parser.parse_args()
    datetime = get_datetime()

    if args.dev:
        beauty_train = pd.read_csv(beauty_dev_repo)
        beauty_val = pd.read_csv(beauty_val_repo)

        fashion_train = pd.read_csv(fashion_dev_repo)
        fashion_val = pd.read_csv(fashion_val_repo)

        mobile_train = pd.read_csv(mobile_dev_repo)
        mobile_val = pd.read_csv(mobile_val_repo)

    else:
        beauty_train = pd.read_csv(beauty_train_repo)
        beauty_val = pd.read_csv(beauty_test_repo)

        fashion_train = pd.read_csv(fashion_train_repo)
        fashion_val = pd.read_csv(fashion_test_repo)

        mobile_train = pd.read_csv(mobile_train_repo)
        mobile_val = pd.read_csv(mobile_test_repo)

    beauty_dict = create_label(beauty_train, beauty_val, 'beauty')
    fashion_dict = create_label(fashion_train, fashion_val, 'fashion')
    mobile_dict = create_label(mobile_train, mobile_val, 'mobile')

    beauty_result_dict = create_prediction(beauty_dict, 'beauty', beauty_topic)
    fashion_result_dict = create_prediction(fashion_dict, 'fashion', fashion_topic)
    mobile_result_dict = create_prediction(mobile_dict, 'mobile', mobile_topic)

    model_path = os.path.join(result_path, 'lgb_{}').format(datetime)
    model_metadata_path = os.path.join(result_metadata_path, 'lgb_{}').format(datetime)

    create_directory(model_path)
    create_directory(model_metadata_path)

    if args.dev:
        joblib.dump(beauty_dict, os.path.join(model_path, 'VAL_beauty_dict.pkl'))
        joblib.dump(fashion_dict, os.path.join(model_path, 'VAL_fashion_dict.pkl'))
        joblib.dump(mobile_dict, os.path.join(model_path, 'VAL_mobile_dict.pkl'))

        joblib.dump(beauty_result_dict, os.path.join(model_path, 'VAL_beauty_result_dict.pkl'))
        joblib.dump(fashion_result_dict, os.path.join(model_path, 'VAL_fashion_result_dict.pkl'))
        joblib.dump(mobile_result_dict, os.path.join(model_path, 'VAL_mobile_result_dict.pkl'))

    else:
        joblib.dump(beauty_dict, os.path.join(model_path, 'beauty_dict.pkl'))
        joblib.dump(fashion_dict, os.path.join(model_path, 'fashion_dict.pkl'))
        joblib.dump(mobile_dict, os.path.join(model_path, 'mobile_dict.pkl'))

        joblib.dump(beauty_result_dict, os.path.join(model_path, 'beauty_result_dict.pkl'))
        joblib.dump(fashion_result_dict, os.path.join(model_path, 'fashion_result_dict.pkl'))
        joblib.dump(mobile_result_dict, os.path.join(model_path, 'mobile_result_dict.pkl'))

    config.save(os.path.join(model_metadata_path, 'config.csv'))
