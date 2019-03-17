import argparse
from sklearn.externals import joblib
from fastai.vision import *

from model.image.fastai.model import fastai_prediction
from model.image.fastai.utils import fix_image_path
from model.common.topic import beauty_columns, mobile_columns, fashion_columns
from utils.envs import *
from utils.logger import logger
from utils.common import get_datetime, create_directory


def main():
    logger.setup_logger('img_fastai')
    datetime = get_datetime()
    model_path = os.path.join(result_path, 'img_fastai_{}').format(datetime)
    model_metadata_path = os.path.join(result_metadata_path, 'img_fastai_{}').format(datetime)

    create_directory(model_path)
    create_directory(model_metadata_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", help='Using Development Dataset and Validation Dataset instead to perform training',
                        action='store_true')
    args = parser.parse_args()

    path = Path(img_root)

    beauty_dev = pd.read_csv(beauty_dev_repo)
    beauty_val = pd.read_csv(beauty_val_repo)

    fashion_dev = pd.read_csv(fashion_dev_repo)
    fashion_val = pd.read_csv(fashion_val_repo)

    mobile_dev = pd.read_csv(mobile_dev_repo)
    mobile_val = pd.read_csv(mobile_val_repo)

    beauty_dev = fix_image_path(beauty_dev, 'beauty_dev')
    beauty_val = fix_image_path(beauty_val, 'beauty_val')

    fashion_dev = fix_image_path(fashion_dev, 'fashion_dev')
    fashion_val = fix_image_path(fashion_val, 'fashion_val')

    mobile_dev = fix_image_path(mobile_dev, 'mobile_dev')
    mobile_val = fix_image_path(mobile_val, 'mobile_val')

    if args.dev:
        train_beauty = beauty_dev
        test_beauty = 'beauty_val'

        train_fashion = fashion_dev
        test_fashion = 'fashion_val'

        train_mobile = mobile_dev
        test_mobile = 'mobile_val'
    else:
        train_beauty = pd.concat([beauty_dev, beauty_val])
        test_beauty = 'beauty_test'

        train_fashion = pd.concat([fashion_dev, fashion_val])
        test_fashion = 'fashion_test'

        train_mobile = pd.concat([mobile_dev, mobile_val])
        test_mobile = 'mobile_test'

    beauty_result_dict = fastai_prediction(train_beauty, test_beauty, beauty_columns, path)
    fashion_result_dict = fastai_prediction(train_fashion, test_fashion, fashion_columns, path)
    mobile_result_dict = fastai_prediction(train_mobile, test_mobile, mobile_columns, path)

    if args.dev:
        joblib.dump(beauty_result_dict, os.path.join(model_path, 'VAL_beauty_result_dict.pkl'))
        joblib.dump(fashion_result_dict, os.path.join(model_path, 'VAL_fashion_result_dict.pkl'))
        joblib.dump(mobile_result_dict, os.path.join(model_path, 'VAL_mobile_result_dict.pkl'))

    else:
        joblib.dump(beauty_result_dict, os.path.join(model_path, 'beauty_result_dict.pkl'))
        joblib.dump(fashion_result_dict, os.path.join(model_path, 'fashion_result_dict.pkl'))
        joblib.dump(mobile_result_dict, os.path.join(model_path, 'mobile_result_dict.pkl'))



