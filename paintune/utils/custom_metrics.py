import re
import numpy as np
from datasets import load_metric

def extract_final_answer(text):
    """최종 답변 추출 함수"""
    # '답:' 뒤의 텍스트 추출
    match = re.search(r'답:\s*(.*?)$', text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return text.strip()

def normalize_math_answer(answer):
    """수학 답변 정규화 (분수, 소수점 등 다양한 표현 통일)"""
    # 공백 제거, 소문자 변환
    answer = re.sub(r'\s+', '', answer).lower()
    
    # 분수 정규화 (예: "3/4" → "0.75")
    fraction_match = re.match(r'^(\d+)/(\d+)$', answer)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        return str(numerator / denominator)
    
    return answer

def math_accuracy_metric(eval_pred):
    """수학 문제에 특화된 평가 지표"""
    predictions, labels = eval_pred
    # 라벨 디코딩은 별도로 수행된다고 가정
    
    prediction_texts = [extract_final_answer(pred) for pred in predictions]
    reference_texts = [extract_final_answer(ref) for ref in labels]
    
    # 정규화된 답변으로 정확도 계산
    normalized_preds = [normalize_math_answer(pred) for pred in prediction_texts]
    normalized_refs = [normalize_math_answer(ref) for ref in reference_texts]
    
    correct = sum(pred == ref for pred, ref in zip(normalized_preds, normalized_refs))
    accuracy = correct / len(normalized_refs)
    
    return {"math_accuracy": accuracy}