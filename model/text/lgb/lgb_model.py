import string

import pandas as pd
import lightgbm as lgb
from category_encoders import OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from model.text.lgb.config import config
from model.text.lgb.eta_zoo import eta_dict
from utils.logger import logger


def text_process(s):
    s = str(s)
    s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()
    return s


def create_label(df, test_df, topic):
    result_dict = {}
    feature_df = df[['title']].copy()
    label_df = df.drop(columns=['itemid', 'title', 'image_path']).copy()

    feature_df['title'] = feature_df['title'].apply(lambda x: text_process(x))
    feature_array = feature_df['title'].values.tolist()
    feature_encoder = TfidfVectorizer()
    feature_encoder.fit(feature_array)
    feature_attr = feature_encoder.transform(feature_array)
    feature_decomposer = TruncatedSVD(config.n_svd)
    feature_decomposer.fit(feature_attr)
    feature_attr = feature_decomposer.transform(feature_attr)

    test_df['title'] = test_df['title'].apply(lambda x: text_process(x))
    test_array = test_df['title'].values.tolist()
    test_attr = feature_encoder.transform(test_array)
    test_attr = feature_decomposer.transform(test_attr)

    train_itemid = df['itemid']
    test_itemid = test_df['itemid']

    result_dict['itemid_train_{}'.format(topic)] = train_itemid
    result_dict['itemid_test_{}'.format(topic)] = test_itemid
    result_dict['X_train_{}'.format(topic)] = feature_attr
    result_dict['X_encoder_{}'.format(topic)] = feature_encoder
    result_dict['X_decomposer_{}'.format(topic)] = feature_decomposer
    result_dict['X_test_{}'.format(topic)] = test_attr

    for column in label_df.columns:
        label_encoder = OrdinalEncoder(cols=[column], handle_unknown='impute')
        label_encoder.fit(label_df[[column]])
        label_attr = label_encoder.transform(label_df[[column]])

        result_dict['Y_train_{}_{}'.format(topic, column)] = label_attr
        result_dict['Y_encoder_{}_{}'.format(topic, column)] = label_encoder
        result_dict['Y_colname_{}_{}'.format(topic, column)] = label_attr.columns

    return result_dict


def create_prediction(topic_dict, topic, column_list):
    result_dict = {}
    for column in column_list:
        logger.info('Prediction on {}, {}'.format(topic, column))
        X_train = topic_dict['X_train_{}'.format(topic)]
        Y_train = topic_dict['Y_train_{}_{}'.format(topic, column)][column].values

        train_idx = Y_train != 0
        X_train = X_train[train_idx]
        Y_train = Y_train[train_idx]

        X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.0001)

        X_test = topic_dict['X_test_{}'.format(topic)]

        ddev = lgb.Dataset(X_dev, label=Y_dev)
        dval = lgb.Dataset(X_val, label=Y_val)
        num_class = Y_train.max() + 1
        #chosen_eta = choose_eta(ddev, dval, num_class)
        chosen_eta = eta_dict['{}_{}'.format(topic, column)]
        logger.info('Chosen eta for {} : {}'.format(column, chosen_eta))

        params = {'objective': 'multiclass',
                  'eta': chosen_eta,
                  'max_depth': 6,
                  'num_leaves': 63,
                  "feature_fraction": 0.7,
                  "bagging_fraction": 0.7,
                  "metric": ['multi_logloss', 'multi_error'],
                  'silent': 1,
                  'nthread': config.n_threads,
                  'num_class': num_class}

        bst = lgb.train(params, ddev, num_boost_round=config.n_round, valid_sets=[ddev, dval],
                        valid_names=['ddev', 'dval'])
        logger.info('{} ddev loss : {}'.format(column, bst.best_score['ddev']['multi_logloss']))
        logger.info('{} ddev error rate : {}'.format(column, bst.best_score['ddev']['multi_error']))
        logger.info('{} dval loss : {}'.format(column, bst.best_score['dval']['multi_logloss']))
        logger.info('{} dval error rate : {}'.format(column, bst.best_score['dval']['multi_error']))
        column_dict = {value2: value1 for value1, value2
                       in topic_dict['Y_encoder_{}_{}'.format(topic, column)].mapping[0]['mapping']}
        column_name = ['NA_VALUE'] + [column_dict[i] for i in range(1, num_class)]
        Y_pred = bst.predict(X_test)
        Y_pred_df = pd.DataFrame(Y_pred, columns=column_name)
        result_dict[column] = Y_pred_df
    return result_dict


