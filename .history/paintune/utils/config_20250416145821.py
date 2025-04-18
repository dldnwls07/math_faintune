# utils/config.py

# 모델 관련 설정값
MODEL_CHECKPOINT = "t5-small"  # "google/flan-t5-base" 등 원하는 모델

# 절대 경로 사용
DATA_PATH = "C:/Users/feca1/Desktop/math_test/your_actual_data_location/hakttong_dataset_fixed.json"

# 또는 상대 경로 수정
DATA_PATH = "../data/hakttong_dataset_fixed.json"

OUTPUT_DIR = "./results"
BATCH_SIZE = 4
NUM_EPOCHS = 3
LOGGING_DIR = "./logs"