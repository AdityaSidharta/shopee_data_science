from tqdm import tqdm
import argparse
from sklearn.externals import joblib
from fastai.vision import *

from model.image.fastai.ml_model import fastai_prediction
from model.common.topic import beauty_columns, mobile_columns, fashion_columns
from utils.envs import *
from utils.logger import logger
from utils.common import get_datetime, create_directory


def fix_image_path(input_df, relative_path):
    df = input_df.copy()
    for idx in tqdm(range(len(input_df))):
        filename = df.at[idx, 'image_path'].split('/')[1]
        if not filename.endswith('.jpg'):
            filename = filename + '.jpg'
        final_filename = os.path.join(relative_path, filename)
        df.at[idx, 'image_path'] = final_filename
    return df


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
    beauty_test = pd.read_csv(beauty_test_repo)

    fashion_dev = pd.read_csv(fashion_dev_repo)
    fashion_val = pd.read_csv(fashion_val_repo)
    fashion_test = pd.read_csv(fashion_test_repo)

    mobile_dev = pd.read_csv(mobile_dev_repo)
    mobile_val = pd.read_csv(mobile_val_repo)
    mobile_test = pd.read_csv(mobile_test_repo)

    beauty_dev = fix_image_path(beauty_dev, 'beauty_dev')
    beauty_val = fix_image_path(beauty_val, 'beauty_val')
    beauty_test = fix_image_path(beauty_test, 'beauty_test')

    fashion_dev = fix_image_path(fashion_dev, 'fashion_dev')
    fashion_val = fix_image_path(fashion_val, 'fashion_val')
    fashion_test = fix_image_path(fashion_test, 'fashion_test')

    mobile_dev = fix_image_path(mobile_dev, 'mobile_dev')
    mobile_val = fix_image_path(mobile_val, 'mobile_val')
    mobile_test = fix_image_path(mobile_test, 'mobile_test')

    if args.dev:
        train_beauty = beauty_dev
        test_beauty = beauty_val
        test_beauty_folder = 'beauty_val'

        train_fashion = fashion_dev
        test_fashion = fashion_val
        test_fashion_folder = 'fashion_val'

        train_mobile = mobile_dev
        test_mobile = mobile_val
        test_mobile_folder = 'mobile_val'
    else:
        train_beauty = pd.concat([beauty_dev, beauty_val])
        test_beauty = beauty_test
        test_beauty_folder = 'beauty_test'

        train_fashion = pd.concat([fashion_dev, fashion_val])
        test_fashion = fashion_test
        test_fashion_folder = 'fashion_test'

        train_mobile = pd.concat([mobile_dev, mobile_val])
        test_mobile = mobile_test
        test_mobile_folder = 'mobile_test'

    beauty_result_dict = fastai_prediction(train_beauty, test_beauty, test_beauty_folder, beauty_columns, path, 'beauty')
    fashion_result_dict = fastai_prediction(train_fashion, test_fashion, test_fashion_folder, fashion_columns, path, 'fashion')
    mobile_result_dict = fastai_prediction(train_mobile, test_mobile, test_mobile_folder, mobile_columns, path, 'mobile')

    if args.dev:
        joblib.dump(beauty_result_dict, os.path.join(model_path, 'VAL_beauty_result_dict.pkl'))
        joblib.dump(fashion_result_dict, os.path.join(model_path, 'VAL_fashion_result_dict.pkl'))
        joblib.dump(mobile_result_dict, os.path.join(model_path, 'VAL_mobile_result_dict.pkl'))

    else:
        joblib.dump(beauty_result_dict, os.path.join(model_path, 'beauty_result_dict.pkl'))
        joblib.dump(fashion_result_dict, os.path.join(model_path, 'fashion_result_dict.pkl'))
        joblib.dump(mobile_result_dict, os.path.join(model_path, 'mobile_result_dict.pkl'))


if __name__ == '__main__':
    main()
