import os

project_path = os.getenv("PROJECT_PATH")

data_source_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, "output")

model_cp_path = os.path.join(output_path, "model_checkpoint")
logger_path = os.path.join(output_path, "logs")
result_path = os.path.join(output_path, "result")

logger_repo = os.path.join(logger_path, "logger.log")
