import os
import utils.api_keys

project_path = os.getenv("PROJECT_PATH")

fono_key = utils.api_keys.FONO_API_KEY
fono_data_path = os.path.join(project_path, "data", "fono_api")

data_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, "output")

model_cp_path = os.path.join(output_path, "model_checkpoint")
logger_path = os.path.join(output_path, "logs")
result_path = os.path.join(output_path, "result")

logger_repo = os.path.join(logger_path, "logger.log")

beauty_image_path = os.path.join(data_path, 'beauty_image')
fashion_image_path = os.path.join(data_path, 'fashion_image')
mobile_image_path = os.path.join(data_path, 'mobile_image')

beauty_train_repo = os.path.join(data_path, 'beauty_data_info_train_competition.csv')
beauty_val_repo = os.path.join(data_path, 'beauty_data_info_val_competition.csv')
beauty_profile_json = os.path.join(data_path, 'beauty_profile_train.json')

fashion_train_repo = os.path.join(data_path, 'fashion_data_info_train_competition.csv')
fashion_val_repo = os.path.join(data_path, 'fashion_data_info_val_competition.csv')
fashion_profile_json = os.path.join(data_path, 'fashion_profile_train.json')

mobile_train_repo = os.path.join(data_path, 'mobile_data_info_train_competition.csv')
mobile_val_repo = os.path.join(data_path, 'mobile_data_info_val_competition.csv')
mobile_profile_json = os.path.join(data_path, 'mobile_profile_train.json')
