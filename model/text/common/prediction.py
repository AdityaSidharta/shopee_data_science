import pandas as pd


def predict_single(Y_pred, topic_dict, topic, column):
        Y_class = Y_pred.argmax(1)
        Y_class_df = pd.DataFrame(Y_class, columns = topic_dict['Y_colname_{}_{}'.format(topic, column)])
        Y_class_result = topic_dict['Y_encoder_{}_{}'.format(topic, column)].inverse_transform(Y_class_df)
        result = pd.DataFrame({
            'id': topic_dict['itemid_test_{}'.format(topic)].apply(lambda x : str(x) + '_{}'.format(column)),
            'tagging': Y_class_result[column].values.astype(int).astype(str)
        })
        return result
