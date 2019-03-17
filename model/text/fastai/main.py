import argparse
from fastai.text import *
from sklearn.externals import joblib

from model.common.topic import beauty_columns, fashion_columns, mobile_columns
from model.text.fastai.class_model import fastai_prediction
from model.text.fastai.lm_model import create_data_lm, create_model_lm, clean_title, load_lm
from utils.common import create_directory, get_datetime
from utils.envs import *
from utils.logger import logger

if __name__ == '__main__':
    logger.setup_logger('pytorch')
    create_directory(os.path.join(result_path, 'static'))
    RESULT_PATH = Path(os.path.join(result_path, 'static'))

    datetime = get_datetime()
    model_path = os.path.join(result_path, 'fastai_{}').format(datetime)
    model_metadata_path = os.path.join(result_metadata_path, 'fastai_{}').format(datetime)

    create_directory(model_path)
    create_directory(model_metadata_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", help='Performing Language Model Training',
                        action='store_true')
    parser.add_argument("--dev", help='Using Development Dataset and Validation Dataset instead to perform training',
                        action='store_true')
    args = parser.parse_args()

    if args.lm:
        beauty_train = pd.read_csv(beauty_train_repo)
        beauty_test = pd.read_csv(beauty_test_repo)

        fashion_train = pd.read_csv(fashion_train_repo)
        fashion_test = pd.read_csv(fashion_test_repo)

        mobile_train = pd.read_csv(mobile_train_repo)
        mobile_test = pd.read_csv(mobile_test_repo)

        kyle_indo_df = pd.read_csv(kyle_indonesia_repo)[['item_name']]
        kyle_sing_df = pd.read_csv(kyle_singapore_repo)[['item_name']]

        for df in [kyle_indo_df, kyle_sing_df]:
            df.item_name = df.item_name.apply(lambda x: clean_title(x))

        kyle_indo_df.columns = ['title']
        kyle_sing_df.columns = ['title']

        full_text_df = pd.concat(
            [beauty_train[['title']],
             beauty_test[['title']],
             fashion_train[['title']],
             fashion_test[['title']],
             mobile_train[['title']],
             mobile_test[['title']],
             kyle_indo_df[['title']],
             kyle_sing_df[['title']]])

        data_lm = create_data_lm(full_text_df)
        learn_lm = create_model_lm(data_lm)

        data_lm.save(RESULT_PATH / 'data_lm.pkl')
        learn_lm.save(RESULT_PATH / 'model_lm.pkl')
        learn_lm.save_encoder(RESULT_PATH / 'encoder_lm.pkl')
    else:
        if args.dev:
            beauty_train = pd.read_csv(beauty_dev_repo)
            beauty_test = pd.read_csv(beauty_val_repo)

            fashion_train = pd.read_csv(fashion_dev_repo)
            fashion_test = pd.read_csv(fashion_val_repo)

            mobile_train = pd.read_csv(mobile_dev_repo)
            mobile_test = pd.read_csv(mobile_val_repo)

        else:
            beauty_train = pd.read_csv(beauty_train_repo)
            beauty_test = pd.read_csv(beauty_test_repo)

            fashion_train = pd.read_csv(fashion_train_repo)
            fashion_test = pd.read_csv(fashion_test_repo)

            mobile_train = pd.read_csv(mobile_train_repo)
            mobile_test = pd.read_csv(mobile_test_repo)

        data_lm = load_data(RESULT_PATH, 'data_lm.pkl')
        learn_lm = load_lm(data_lm, RESULT_PATH)

        beauty_result_dict = fastai_prediction(beauty_train, beauty_test, beauty_columns, data_lm, RESULT_PATH)
        fashion_result_dict = fastai_prediction(fashion_train, fashion_test, fashion_columns, data_lm, RESULT_PATH)
        mobile_result_dict = fastai_prediction(mobile_train, mobile_test, mobile_columns, data_lm, RESULT_PATH)

        if args.dev:
            joblib.dump(beauty_result_dict, os.path.join(model_path, 'VAL_beauty_result_dict.pkl'))
            joblib.dump(fashion_result_dict, os.path.join(model_path, 'VAL_fashion_result_dict.pkl'))
            joblib.dump(mobile_result_dict, os.path.join(model_path, 'VAL_mobile_result_dict.pkl'))

        else:
            joblib.dump(beauty_result_dict, os.path.join(model_path, 'beauty_result_dict.pkl'))
            joblib.dump(fashion_result_dict, os.path.join(model_path, 'fashion_result_dict.pkl'))
            joblib.dump(mobile_result_dict, os.path.join(model_path, 'mobile_result_dict.pkl'))
