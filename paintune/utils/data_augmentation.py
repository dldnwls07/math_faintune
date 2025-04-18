import random
import json
from copy import deepcopy

def augment_math_dataset(dataset_path, output_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    augmented_data = []
    for item in data:
        augmented_data.append(item)  # 원본 추가
        
        # 1. 숫자 변형 (확률 문제에서 숫자만 변경)
        if "동전" in item["question"] or "주사위" in item["question"]:
            modified_item = deepcopy(item)
            # 숫자 변경 로직 (예: '3번' → '4번')
            modified_item["question"] = modified_item["question"].replace("3번", "4번")
            # 답변도 적절히 수정 (실제로는 더 복잡한 로직 필요)
            augmented_data.append(modified_item)
        
        # 2. 문제 표현 방식 변경
        if "확률" in item["question"]:
            rephrased_item = deepcopy(item)
            rephrased_item["question"] = rephrased_item["question"].replace(
                "확률은?", "확률을 구하시오."
            )
            augmented_data.append(rephrased_item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)

# 사용 예: augment_math_dataset("hakttong_dataset_fixed.json", "hakttong_augmented.json")