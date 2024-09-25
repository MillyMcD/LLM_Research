import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def sem_score(a,b,func=None,embedding_model:str="all-mpnet-base-v2",cache_dir = '/data/cache'):
    if func is None:
        func = HuggingFaceEmbeddings(
            model_name =  f'sentence-transformers/{embedding_model}',
            cache_folder = cache_dir
        )

    va = func.embed_query(a)
    vb = func.embed_query(b)
    return cosine_similarity(va,vb)