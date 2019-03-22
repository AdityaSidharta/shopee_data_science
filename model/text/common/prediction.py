import pandas as pd
import numpy as np
from model.common.topic import beauty_columns, fashion_columns, mobile_columns

concat_ratio_dict = {'Beauty_Colour_group': {'fastai': 0.6, 'lgb': 0.4},
 'Beauty_Brand': {'fastai': 0.5, 'lgb': 0.5},
 'Beauty_Benefits': {'fastai': 0.4, 'lgb': 0.6},
 'Beauty_Product_texture': {'fastai': 0.5, 'lgb': 0.5},
 'Beauty_Skin_type': {'fastai': 0.5, 'lgb': 0.5},
 'Fashion_Pattern': {'fastai': 0.5, 'lgb': 0.5},
 'Fashion_Collar Type': {'fastai': 0.9, 'lgb': 0.1},
 'Fashion_Sleeves': {'fastai': 0.5, 'lgb': 0.5},
 'Fashion_Fashion Trend': {'fastai': 0.5, 'lgb': 0.5},
 'Fashion_Clothing Material': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Operating System': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Features': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Network Connections': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Memory RAM': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Brand': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Warranty Period': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Storage Capacity': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Color Family': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Phone Model': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Camera': {'fastai': 0.5, 'lgb': 0.5},
 'Mobile_Phone Screen Size': {'fastai': 0.5, 'lgb': 0.5}}


def concat_prediction(lgb_topic_result_dict, fastai_topic_result_dict):
    result_dict = {}
    for column in fastai_topic_result_dict.keys():
        fastai_df = fastai_topic_result_dict[column]
        lgb_df = lgb_topic_result_dict[column]

        fastai_df.columns = [str(x) for x in fastai_df.columns]
        lgb_df.columns = [str(x) for x in lgb_df.columns]

        column_list = fastai_df.columns.tolist()
        lgb_df = lgb_df[column_list]

        fastai_array = fastai_df.values
        lgb_array = lgb_df.values

        result_array =(0.5 * fastai_array) + (0.5 * lgb_array)
        result_dict[column] = pd.DataFrame(result_array, columns=column_list)
    return result_dict


def concat_prediction_image(lgb_topic_result_dict, fastai_topic_result_dict, image_topic_result_dict):
    result_dict = {}
    fastai_ratio = 0.49
    lgb_ratio = 0.01
    image_ratio = 0.50

    print("fastai_ratio : {}".format(fastai_ratio))
    print("lgb_ratio : {}".format(lgb_ratio))
    print("image_ratio : {}".format(image_ratio))

    for column in fastai_topic_result_dict.keys():
        fastai_df = fastai_topic_result_dict[column]
        lgb_df = lgb_topic_result_dict[column]
        image_df = image_topic_result_dict[column]

        fastai_df.columns = [str(x) for x in fastai_df.columns]
        lgb_df.columns = [str(x) for x in lgb_df.columns]
        image_df.columns = [str(x) for x in image_df.columns]

        column_list = fastai_df.columns.tolist()
        lgb_df = lgb_df[column_list]

        diff_keys = set(column_list) - set(image_df.columns)
        for diff_column in diff_keys:
            image_df[diff_column] = 0.0
        image_df = image_df[column_list]

        fastai_array = fastai_df.values
        lgb_array = lgb_df.values
        image_array = image_df.values

        result_array = (fastai_ratio * fastai_array) + (lgb_ratio * lgb_array) + (image_ratio * image_array)
        result_dict[column] = pd.DataFrame(result_array, columns=column_list)
    return result_dict


def concat_prediction_ratio(lgb_topic_result_dict, fastai_topic_result_dict, topic):
    result_dict = {}
    for column in fastai_topic_result_dict.keys():
        fastai_df = fastai_topic_result_dict[column]
        lgb_df = lgb_topic_result_dict[column]

        fastai_df.columns = [str(x) for x in fastai_df.columns]
        lgb_df.columns = [str(x) for x in lgb_df.columns]

        column_list = fastai_df.columns.tolist()
        lgb_df = lgb_df[column_list]

        fastai_array = fastai_df.values
        lgb_array = lgb_df.values

        fastai_ratio = concat_ratio_dict['{}_{}'.format(topic, column)]['fastai']
        lgb_ratio = concat_ratio_dict['{}_{}'.format(topic, column)]['lgb']

        assert fastai_ratio + lgb_ratio == 1

        result_array =(fastai_ratio * fastai_array) + (lgb_ratio * lgb_array)
        result_dict[column] = pd.DataFrame(result_array, columns=column_list)
    return result_dict


def predict_single(topic_test_df, topic_result_dict):
    result_list = []
    for column in topic_result_dict.keys():
        raw_df = topic_result_dict[column]
        raw_df_columns = raw_df.columns
        raw_df_coldict = {key: str(value).split('.')[0] for key, value in enumerate(raw_df_columns)}
        result_array = raw_df.values.argmax(1)
        result = np.vectorize(raw_df_coldict.get)(result_array)
        result_df = pd.DataFrame({
            'id': topic_test_df['itemid'].apply(lambda x: str(x) + '_{}'.format(column)),
            'tagging': result
        })
        result_list.append(result_df)
    return pd.concat(result_list)


def predict_double(topic_test_df, topic_result_dict):
    result_list = []
    itemid_list = topic_test_df['itemid'].values.tolist()
    for column in topic_result_dict.keys():
        raw_df = topic_result_dict[column]
        value = raw_df.values
        idx_argsort = value.argsort()[:, ::-1]
        sorted_value = value.copy()
        sorted_value.sort()
        raw_df_columns = raw_df.columns
        raw_df_coldict = {key: str(value).split('.')[0] for key, value in enumerate(raw_df_columns)}

        for idx in range(len(itemid_list)):
            preds = idx_argsort[idx, :2]
            result = np.vectorize(raw_df_coldict.get)(preds)
            result_list.append({
                'id': str(itemid_list[idx]) + '_{}'.format(column),
                'tagging': ' '.join(result.tolist())
            })
    return pd.DataFrame(result_list)


def predict_threshold(topic_test_df, topic_result_dict, threshold):
    result_list = []
    itemid_list = topic_test_df['itemid'].values.tolist()
    for column in topic_result_dict.keys():
        raw_df = topic_result_dict[column]
        value = raw_df.values
        idx_argsort = value.argsort()[:, ::-1]
        sorted_value = value.copy()
        sorted_value.sort()
        sorted_value = sorted_value[:, ::-1]
        cumsum_value = np.cumsum(sorted_value, axis=1)
        raw_df_columns = raw_df.columns
        raw_df_coldict = {key: str(value).split('.')[0] for key, value in enumerate(raw_df_columns)}

        for idx in range(len(itemid_list)):
            n_preds = min(np.searchsorted(cumsum_value[idx], threshold), 4)
            preds = idx_argsort[idx, :n_preds + 1]
            result = np.vectorize(raw_df_coldict.get)(preds)
            result_list.append({
                'id': str(itemid_list[idx]) + '_{}'.format(column),
                'tagging': ' '.join(result.tolist())
            })
    return pd.DataFrame(result_list)


def build_prediction_list(beauty_test_df, fashion_test_df, mobile_test_df):
    prediction_list = []
    for itemid in beauty_test_df['itemid'].values.tolist():
        for column in beauty_columns:
            prediction_list.append(str(itemid) + '_{}'.format(column))
    for itemid in fashion_test_df['itemid'].values.tolist():
        for column in fashion_columns:
            prediction_list.append(str(itemid) + '_{}'.format(column))
    for itemid in mobile_test_df['itemid'].values.tolist():
        for column in mobile_columns:
            prediction_list.append(str(itemid) + '_{}'.format(column))
    return pd.DataFrame({
        'id': prediction_list
    })


def concat_submission(beauty_submission_df, fashion_submission_df, mobile_submission_df,
                      beauty_test_df, fashion_test_df, mobile_test_df):
    prediction_df = build_prediction_list(beauty_test_df, fashion_test_df, mobile_test_df)
    answer_df = pd.concat([beauty_submission_df, fashion_submission_df, mobile_submission_df])
    assert len(answer_df) == len(prediction_df)
    submission_df = prediction_df[['id']].merge(answer_df, on='id', how='left')
    return submission_df
