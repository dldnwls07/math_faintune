from transformers import AutoTokenizer

def load_tokenizer(model_checkpoint):
    return AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(example, tokenizer):
    """
    example: JSON의 한 row 예시
    {
      "question": "어떤 주사위를 두 번 던질 때, 합이 7이 될 확률은?",
      "answer": "1/6",
      ...
    }
    """
    input_text = "문제: " + example["question"]
    target_text = "답: " + example["answer"]
    
    model_inputs = tokenizer(
        input_text,
        max_length=128,
        truncation=True
    )
    labels = tokenizer(
        target_text,
        max_length=32,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
