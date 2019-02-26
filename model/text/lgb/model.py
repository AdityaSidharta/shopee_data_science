from utils.envs import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.externals import joblib
import datetime
import string


def text_process(s):
    s = str(s)
    s.translate(str.maketrans('','',string.punctuation))
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
    feature_decomposer = TruncatedSVD(500)
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
        X_train = topic_dict['X_train_{}'.format(topic)]
        Y_train = topic_dict['Y_train_{}_{}'.format(topic, column)][column].values

        train_idx = Y_train != 0
        X_train = X_train[train_idx]
        Y_train = Y_train[train_idx]

        X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

        X_test = topic_dict['X_test_{}'.format(topic)]

        ddev = lgb.Dataset(X_dev, label=Y_dev)
        dval = lgb.Dataset(X_val, label=Y_val)

        params = {'objective': 'multiclass',
                  'eta': 0.01,
                  'max_depth': 6,
                  'num_leaves': 63,
                  "feature_fraction": 0.7,
                  "bagging_fraction": 0.7,
                  'silent': 1,
                  'nthread': 6,
                  'num_class': Y_train.max() + 1}

        bst = lgb.train(params, ddev, num_boost_round=2000, valid_sets=[ddev, dval], valid_names=['ddev', 'dval'],
                        early_stopping_rounds=50)
        Y_pred = bst.predict(X_test)
        result_dict['Y_pred_{}_{}'.format(topic, column)] = Y_pred

    #        Y_class = Y_pred.argmax(1)
    #        Y_class_df = pd.DataFrame(Y_class, columns = topic_dict['Y_colname_{}_{}'.format(topic, column)])
    #        Y_class_result = topic_dict['Y_encoder_{}_{}'.format(topic, column)].inverse_transform(Y_class_df)
    #        result = pd.DataFrame({
    #            'id': topic_dict['itemid_test_{}'.format(topic)].apply(lambda x : str(x) + '_{}'.format(column)),
    #            'tagging': Y_class_result[column].values.astype(int).astype(str)
    #        })
    #        result_dict['result_df_{}_{}'.format(topic, column)] = result
    return result_dict
