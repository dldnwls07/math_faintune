# train.py

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
# 상대 경로로 수정
from .config import (
    MODEL_CHECKPOINT, DATA_PATH, OUTPUT_DIR, BATCH_SIZE, NUM_EPOCHS, LOGGING_DIR
)
from .preproce import load_tokenizer, preprocess_function

def main():
    # 1) 데이터셋 로드
    data_files = {"train": DATA_PATH}
    dataset = load_dataset("json", data_files=data_files)

    # 2) 토크나이저 로드
    tokenizer = load_tokenizer(MODEL_CHECKPOINT)

    # 3) 전처리
    tokenized_dataset = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=False  # 문제 많으면 True로
    )

    # 4) 모델 로드
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # 5) 학습 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        logging_dir=LOGGING_DIR,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2
    )

    # 6) Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # 7) 학습 시작
    print("학습을 시작합니다...")
    trainer.train()
    print("학습이 완료되었습니다!")

    # 8) 학습된 모델 저장
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
