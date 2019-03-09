import pandas as pd
import numpy as np
from model.text.common.topic import beauty_columns, fashion_columns, mobile_columns

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
            n_preds = np.searchsorted(cumsum_value[idx], threshold)
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
