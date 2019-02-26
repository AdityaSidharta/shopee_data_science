import pandas as pd
from utils.envs import *
from sklearn.externals import joblib
from utils.common import create_directory, get_datetime
from model.text.lgb.model import create_label, create_prediction

beauty_topic = ['Colour_group', 'Brand', 'Benefits', 'Product_texture', 'Skin_type']
fashion_topic =['Pattern', 'Collar Type', 'Sleeves', 'Fashion Trend',
                'Clothing Material']
mobile_topic = ['Operating System', 'Features',
               'Network Connections', 'Memory RAM', 'Brand', 'Warranty Period',
               'Storage Capacity', 'Color Family', 'Phone Model', 'Camera',
               'Phone Screen Size']


if __name__ == '__main__':
    beauty_train = pd.read_csv(beauty_train_repo)
    beauty_val = pd.read_csv(beauty_val_repo)

    fashion_train = pd.read_csv(fashion_train_repo)
    fashion_val = pd.read_csv(fashion_val_repo)

    mobile_train = pd.read_csv(mobile_train_repo)
    mobile_val = pd.read_csv(mobile_val_repo)

    beauty_dict = create_label(beauty_train, beauty_val, 'beauty')
    fashion_dict = create_label(fashion_train, fashion_val, 'fashion')
    mobile_dict = create_label(mobile_train, mobile_val, 'mobile')

    beauty_result_dict = create_prediction(beauty_dict, 'beauty', beauty_topic)
    fashion_result_dict = create_prediction(fashion_dict, 'fashion', fashion_topic)
    mobile_result_dict = create_prediction(mobile_dict, 'mobile', mobile_topic)

    datetime = get_datetime()
    model_path = os.path.join(result_path, 'lgb_{}').format(datetime)
    model_metadata_path = os.path.join(result_metadata_path, 'lgb_{}').format(datetime)

    create_directory(model_path)
    create_directory(model_metadata_path)

