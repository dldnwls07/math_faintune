# app.py
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnablePassthrough
import torch
from transformers import pipeline

# 학습된 모델 로드
model_path = "./results_math_specialized"
math_pipeline = pipeline(
    "text2text-generation", 
    model=model_path,
    device=0 if torch.cuda.is_available() else -1,  # GPU 사용
    max_length=512
)

# 모델 래핑
def process_math_query(query):
    # 입력 전처리
    if "확률" in query.lower():
        prompt = f"확률 문제: {query} 확률 공식을 이용하여 풀이하시오."
    else:
        prompt = f"수학 문제: {query}"
    
    # 모델 추론
    result = math_pipeline(prompt)[0]["generated_text"]
    
    # 출력 후처리
    return {"problem": query, "solution": result}

# FastAPI 앱 생성
app = FastAPI(title="확률통계 문제 해결기")

# Langserve 라우트 추가
add_routes(
    app,
    RunnablePassthrough(process_math_query),
    path="/math-solver",
    input_type=str,  # 입력은 문자열 (수학 문제)
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)