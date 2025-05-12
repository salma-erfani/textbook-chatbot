import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


def load_model_and_chunks(load_generation=False):
    # load models
    embedding_model_name = "HooshvareLab/bert-fa-base-uncased"
    generation_model_name = "SajjadAyoubi/bert-base-fa-qa"
    
    embedding = Transformer(embedding_model_name)
    pooling = Pooling(embedding.get_word_embedding_dimension())
    embedding_model = SentenceTransformer(modules=[embedding, pooling])
    
    if load_generation:
        generation_model = pipeline(
			"question-answering",
			model=generation_model_name,
			tokenizer=generation_model_name
		)
    
    # Load chunks
    with open('textbook_chunks_with_embeddings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for chunk in data['chunks']:
            chunk['embedding'] = np.array(chunk['embedding'])
    
    if load_generation:
        return embedding_model, generation_model, data['chunks']
    return embedding_model, data['chunks']

  
def find_relevant_chunks(question, chunks, model, top_k=3):
	question_embedding = model.encode(question)

	scored_chunks = []
	for chunk in chunks:
		chunk_embedding = np.array(chunk['embedding'])
		similarity = cosine_similarity([question_embedding], [chunk_embedding])[0][0]
		scored_chunks.append({
     		**chunk,
     		"similarity_score": float(similarity)
    	})

	scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
	return scored_chunks[:top_k]



def find_answer_from_data(question, chunks, model, top_k=3):
	relevant_chunks = find_relevant_chunks(question, chunks, model, top_k=top_k)
	results = []
	for ch in relevant_chunks:
		results.append({
			'similarity score': ch['similarity_score'],
			'source page': ch['page'],
			'source heading': ch['heading'],
			'source text': ch['text']
		})
	return results


def generate_answer(question, context, qa_model):
	result = qa_model(question=question, context=context)
	return {
		"answer": result['answer'],
		"confidence": result['score'],
    	"source": context
	}
  
  
def generate_answer_with_context(question, embedding_model, generation_model, chunks, top_contexts=5):
	relevant_chunks = find_relevant_chunks(question, chunks, embedding_model, top_k=top_contexts)
	context = '\n'.join([item['text'] for item in relevant_chunks])
	answer = generate_answer(question, context, generation_model)
	return answer, context, relevant_chunks


def get_reliable_answer(question, embedding_model, generation_model, chunks, min_confidence=0.3):
	answer, _, relevant_chunks = generate_answer_with_context(
									question,
									embedding_model,
									generation_model,
									chunks
								)
	if answer and answer['confidence'] >= min_confidence:
		return answer
	else:
		# best matching chunk
		best_chunk = relevant_chunks[0]
		return {
			"answer": best_chunk['text'],
			"confidence": 0.99,
			"source": best_chunk.get('heading'),
			"fallback": True
		}