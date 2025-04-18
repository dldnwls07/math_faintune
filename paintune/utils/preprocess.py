from transformers import AutoTokenizer

def load_tokenizer(model_checkpoint):
    return AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples, tokenizer):
    """확률통계 문제에 특화된 전처리 함수"""
    input_texts = []
    target_texts = []
    
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]
        
        # 리스트 처리
        if isinstance(question, list): question = question[0] if question else ""
        if isinstance(answer, list): answer = answer[0] if answer else ""
        
    
        # 문제 유형 감지 및 특화된 프롬프트 생성
        if any(term in question.lower() for term in ["확률", "probability", "경우의 수"]):
            input_text = "확률 문제: " + str(question) + " 확률 공식을 이용하여 풀이하시오."
        elif any(term in question.lower() for term in ["통계", "statistics", "평균", "분산"]):
            input_text = "통계 문제: " + str(question) + " 통계 개념을 이용하여 풀이하시오."
        else:
            input_text = "수학 문제: " + str(question)
            
        # 단계별 풀이 유도 (Chain-of-Thought)
        target_text = "풀이 과정: 이 문제를 단계별로 접근해보면...\n답: " + str(answer)
        
        input_texts.append(input_text)
        target_texts.append(target_text)
    
    model_inputs = tokenizer(
        input_texts,
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    
    labels = tokenizer(
        target_texts,
        max_length=32,
        truncation=True,
        padding="max_length",
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
