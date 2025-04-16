# train.py

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
# 상대 경로로 변경하여 직접 실행 시 발생하는 문제 해결
from utils.config import (
    MODEL_CHECKPOINT, DATA_PATH, OUTPUT_DIR, BATCH_SIZE, NUM_EPOCHS, LOGGING_DIR
)
from utils.preprocess import load_tokenizer, preprocess_function

def main():
    try:
        # 1) 데이터셋 로드
        data_files = {"train": DATA_PATH}
        dataset = load_dataset("json", data_files=data_files)

        # 데이터셋 분할
        train_test_split = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

        # 2) 토크나이저 로드
        tokenizer = load_tokenizer(MODEL_CHECKPOINT)

        # 3) 전처리
        train_tokenized = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True  # 대규모 데이터셋에 더 효율적
        )
        eval_tokenized = eval_dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True
        )

        # 4) 모델 로드
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

        # 5) 학습 설정
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,  # 평가용 배치 크기
            num_train_epochs=NUM_EPOCHS,
            logging_dir=LOGGING_DIR,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="epoch",  # 각 epoch마다 평가
            load_best_model_at_end=True,  # 최고 성능 모델 불러오기
            metric_for_best_model="eval_loss",  # 최고 성능 기준
            fp16=True,  # 16비트 정밀도로 학습 (GPU에서 속도 향상)
        )

        # 6) Trainer 생성
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,  # 검증 데이터셋 추가
            tokenizer=tokenizer
        )

        # 7) 학습 시작
        print("학습을 시작합니다...")
        trainer.train()
        print("학습이 완료되었습니다!")

        # 8) 학습된 모델 저장
        trainer.save_model(OUTPUT_DIR)
    except Exception as e:
        print(f"학습 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
