import pandas as pd
import numpy as np
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from sentence_transformers import CrossEncoder

class ChromaDB:
  """
  Vector Database via ChromaDB
  """
  def __init__(self,cache_dir,data_df=None, 
               embedding_model:str = "all-mpnet-base-v2"):
    self.cache_dir = cache_dir
    self.data_df = data_df
    
    #create embedding function
  
    self.embedding_model = f'sentence-transformers/{embedding_model}'
    self.embedding_function = HuggingFaceEmbeddings(
        model_name=self.embedding_model,
        cache_folder=str(self.cache_dir)+'/huggingface_cache'
    )

    self.reranker = CrossEncoder(
      model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
      max_length = 512 #response
    )

    self.chromadb_dir = Path(self.cache_dir)/'chromadb'
    
    #create a vector store
    self.vector_store = Chroma(
        persist_directory = str(self.chromadb_dir),
        embedding_function = self.embedding_function
    )

    #only ingest if documents dont exist
    if len(self.vector_store.get()['documents']) == 0:
      if self.data_df is not None:    
        self.ingest_df(data_df)

  @staticmethod
  def metadata_func(record):
    """
    Create a record
    """
    keys = ['id','context','response']
    return {k:record[k] for k in keys if k in record}
  
  def ingest_df(self,df):
    """
    ingest a dataframe into chromadb
    """
    records = df.to_dict(orient='records')

    docs = []
    for record in records:
      raw_doc = Document(page_content=record['question'],
                          metadata = self.metadata_func(record))
      docs.append(raw_doc)
    
    chroma = Chroma.from_documents(docs,self.embedding_function,
                                   persist_directory=str(self.chromadb_dir))
    
    self.vector_store._collection.add(
        embeddings = chroma.get()['embeddings'],
        metadatas = chroma.get()['metadatas'],
        documents = chroma.get()['documents'],
        ids = chroma.get()['ids']
    )
    self.vector_store.persist()
  
  def rerank(self,query,retrieval):
    """
    Rerank the documents using CrossEncoder model
    """
    scores = self.reranker.predict(
      [(query,ret[0].page_content) for ret in retrieval]
    )
    return [retrieval[i] for i in np.argsort(scores)[::-1]][:3]
  
  def retrieve(self,query,k=4,key:str='response',as_prompt:bool=False,
               advanced:bool=False):
    """
    retrieve top k documents
    """
    if advanced:
      k = 10
    retrieval = self.vector_store.similarity_search_with_relevance_scores(
        query,k)

    if advanced:
      retrieval = self.rerank(query,retrieval)
  
    results = []
    for r in retrieval:
      results.append(r[0].metadata[key])
    if not as_prompt:
      return results
    
    prompt = 'Use the following information to generate your answer:\n'
    for r in results:
      prompt += f'- {r}\n'
    return prompt