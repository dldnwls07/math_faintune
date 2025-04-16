from transformers import AutoTokenizer

def load_tokenizer(model_checkpoint):
    return AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples, tokenizer):
    """
    배치 처리를 위한 전처리 함수
    """
    input_texts = []
    target_texts = []
    
    for i in range(len(examples["question"])):
        # 리스트인 경우 첫 번째 항목 또는 문자열로 변환
        question = examples["question"][i]
        answer = examples["answer"][i]
        
        # 리스트 처리
        if isinstance(question, list):
            question = question[0] if question else ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
            
        input_text = "문제: " + str(question)
        target_text = "답: " + str(answer)
        
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
