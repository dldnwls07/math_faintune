# utils/config.py

# 모델 관련 설정값
MODEL_CHECKPOINT = "google/flan-t5-base"  # 더 큰 모델로 업그레이드

# 절대 경로 사용 (하나만 유지하고 나머지는 주석 처리)
DATA_PATH = r"C:\Users\user\Desktop\math_faintune\paintune\data\hakttong_dataset_fixed.json"

OUTPUT_DIR = "./results_math_specialized"
BATCH_SIZE = 6  # GPU 메모리에 맞게 조정
NUM_EPOCHS = 5  # 더 많은 에폭
LEARNING_RATE = 2e-5  # 더 작은 학습률로 세밀한 학습
WEIGHT_DECAY = 0.01  # 과적합 방지
LOGGING_DIR = "./logs"