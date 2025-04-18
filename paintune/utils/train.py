import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from paintune.utils.config import (
    MODEL_CHECKPOINT, DATA_PATH, OUTPUT_DIR, BATCH_SIZE, NUM_EPOCHS, LOGGING_DIR
)
from paintune.utils.preprocess import load_tokenizer, preprocess_function
from peft import get_peft_model, LoraConfig, TaskType

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

        # LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,  # LoRA의 랭크
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]  # T5 모델의 Attention 모듈
        )

        # LoRA 모델로 변환
        model = get_peft_model(model, peft_config)

        # 학습 가능한 파라미터만 표시
        print(f"학습 가능한 파라미터: {model.print_trainable_parameters()}")

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
            evaluation_strategy="epoch",  # 평가 전략
            load_best_model_at_end=True,  # 학습 종료 후 가장 좋은 모델 로드
            metric_for_best_model="eval_loss",  # 평가 지표
            fp16=True,  # 16비트 정밀도로 학습 (GPU에서 속도 향상)
            do_eval=True,
            eval_steps=500,  # 평가 주기
        )

        # 6) Trainer 생성
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized, 
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
