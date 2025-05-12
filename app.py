from fastapi import FastAPI
from model_loader import get_reliable_answer, load_model_and_chunks, find_answer_from_data


app = FastAPI()
embedding_model, chunks = load_model_and_chunks()
# for generation
# embedding_model, generation_model, chunks = load_model_and_chunks()


@app.post("/ask")
async def ask_question(question: str):
    # 1. answer from relevant data
    answers = find_answer_from_data(question, chunks, embedding_model, top_k=5)
    return answers
    
    # 2. generation with llm
    # answer = get_reliable_answer(
    #     question,
    #     embedding_model,
    #     generation_model,
    #     chunks
    # )
    # return answer
    
    
    