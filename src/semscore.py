import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

def cosine_similarity(a:list,b:list):
    """
    Compute the cosine similarity between two vectors

    Parameters
    ----------
    a : `list`
        first vector
    b : `list`
        second vector

    Returns
    -------
    `float`
        Cosine similarity value
    """
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def sem_score(a:list,b:list,func=None,embedding_model:str="all-mpnet-base-v2",cache_dir = '/data/cache'):
    """
    Compute the SemScore, which is basically just the cosine sim between two text embeddings.

    Parameters
    ----------
    a : `list`
        first vector
    b : `list`
        second vector
    func :
        An embedding function
    embedding_model : `str`
        huggingface model
    cache_dir : `str`
        place to store weights

    Returns
    -------
    `float`
        Cosine similarity value
    """
    if func is None:
        func = HuggingFaceEmbeddings(
            model_name =  f'sentence-transformers/{embedding_model}',
            cache_folder = cache_dir
        )

    va = func.embed_query(a)
    vb = func.embed_query(b)
    return cosine_similarity(va,vb)