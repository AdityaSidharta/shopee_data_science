import os
import utils.api_keys

project_path = os.getenv("PROJECT_PATH")

fono_key = utils.api_keys.FONO_API_KEY
fono_data_path = os.path.join(project_path, "data", "fono_api")

data_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, "output")

device_train_ct = os.path.join(data_path, "phone_counts.json")
not_trained_devices = os.path.join(data_path, "not_trained_devices.json")
gsm_arena = os.path.join(data_path, "gsmarena.json")

model_cp_path = os.path.join(output_path, "model_checkpoint")
logger_path = os.path.join(output_path, "logs")
result_path = os.path.join(output_path, "result")
result_metadata_path = os.path.join(output_path, 'result_metadata')

beauty_image_path = os.path.join(data_path, 'beauty_image')
fashion_image_path = os.path.join(data_path, 'fashion_image')
mobile_image_path = os.path.join(data_path, 'mobile_image')

sample_submission_repo = os.path.join(data_path, 'data_info_val_sample_submission.csv')

beauty_train_repo = os.path.join(data_path, 'beauty_data_info_train_competition.csv')
beauty_dev_repo = os.path.join(data_path, 'beauty_data_info_dev_competition.csv')
beauty_val_repo = os.path.join(data_path, 'beauty_data_info_val_competition.csv')
beauty_test_repo = os.path.join(data_path, 'beauty_data_info_test_competition.csv')
beauty_profile_json = os.path.join(data_path, 'beauty_profile_train.json')

fashion_train_repo = os.path.join(data_path, 'fashion_data_info_train_competition.csv')
fashion_dev_repo = os.path.join(data_path, 'fashion_data_info_dev_competition.csv')
fashion_val_repo = os.path.join(data_path, 'fashion_data_info_val_competition.csv')
fashion_test_repo = os.path.join(data_path, 'fashion_data_info_test_competition.csv')
fashion_profile_json = os.path.join(data_path, 'fashion_profile_train.json')

mobile_train_repo = os.path.join(data_path, 'mobile_data_info_train_competition.csv')
mobile_dev_repo = os.path.join(data_path, 'mobile_data_info_dev_competition.csv')
mobile_val_repo = os.path.join(data_path, 'mobile_data_info_val_competition.csv')
mobile_test_repo = os.path.join(data_path, 'mobile_data_info_test_competition.csv')
mobile_profile_json = os.path.join(data_path, 'mobile_profile_train.json')

kyle_singapore_repo = os.path.join(data_path, 'kyle_singapore.csv')
kyle_indonesia_repo = os.path.join(data_path, 'kyle_indonesia.csv')

img_root = '/home/adityasidharta/git/shopee_data'

img_beauty_dev_folder = os.path.join(img_root, 'beauty_dev')
img_beauty_val_folder = os.path.join(img_root, 'beauty_val')
img_beauty_test_folder = os.path.join(img_root, 'beauty_test')

img_fashion_dev_folder = os.path.join(img_root, 'fashion_dev')
img_fashion_val_folder = os.path.join(img_root, 'fashion_val')
img_fashion_test_folder = os.path.join(img_root, 'fashion_test')

img_mobile_dev_folder = os.path.join(img_root, 'mobile_dev')
img_mobile_val_folder = os.path.join(img_root, 'mobile_val')
img_mobile_test_folder = os.path.join(img_root, 'mobile_test')
