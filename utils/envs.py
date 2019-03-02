import os

project_path = os.getenv("PROJECT_PATH")

data_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, "output")

model_cp_path = os.path.join(output_path, "model_checkpoint")
logger_path = os.path.join(output_path, "logs")
result_path = os.path.join(output_path, "result")
result_metadata_path = os.path.join(output_path, 'result_metadata')

beauty_image_path = os.path.join(data_path, 'beauty_image')
fashion_image_path = os.path.join(data_path, 'fashion_image')
mobile_image_path = os.path.join(data_path, 'mobile_image')

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