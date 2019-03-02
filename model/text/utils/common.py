import datetime
import string

from category_encoders import OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def write_log(path, text):
    with open(path, 'a+') as f:
        f.write('{}_{}\n'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), text))


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
