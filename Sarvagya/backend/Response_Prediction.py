from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Model import createQueryResponseEmbedding

def cosineSimilarity(embedding1,embedding2):
  embedding1 = embedding1.reshape(1, -1)
  embedding2 = embedding2.reshape(1, -1)
  similarity = cosine_similarity(embedding1, embedding2)[0, 0]
  return similarity

def getMostPersonalizedResponse(query ,responses, user_embedding):
  response_similarities = []
  for response in responses:
    temp_embedding = createQueryResponseEmbedding(query, response)
    response_similarities.append(cosineSimilarity(user_embedding, temp_embedding))
  
  response_similarities = np.array(response_similarities)
  return responses[np.argmax(response_similarities)]

