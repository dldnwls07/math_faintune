# utils/infer.py

import torch
from transformers import AutoModelForSeq2SeqLM
# utils를 상대 경로로 변경
from .config import MODEL_CHECKPOINT, OUTPUT_DIR
from .preproce import load_tokenizer

def load_trained_model_and_tokenizer():
    # 학습된 모델 불러오기
    tokenizer = load_tokenizer(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    return model, tokenizer

def predict(model, tokenizer, question):
    input_text = "문제: " + question
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids, max_length=32)
    # 디코딩
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

if __name__ == "__main__":
    # 간단한 테스트
    model, tokenizer = load_trained_model_and_tokenizer()
    test_question = "동전을 5번 던질 때, 앞면이 3번 나올 확률은?"
    result = predict(model, tokenizer, test_question)
    print("문제:", test_question)
    print("예측 답:", result)
