import pandas as pd
import numpy as np
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from sentence_transformers import CrossEncoder

class ChromaDB:
  """
  Vector Database. A wrapper around Chroma

  Parameters
  ----------
  cache_dir : `str`
      place to store the vector database
  data_df : `pd.DataFrame`
      dataframe containing data to ingest
  embedding_model : `str`
      Chosen embedding model
  """
  def __init__(self,cache_dir:str,data_df:pd.DataFrame=None, embedding_model:str = "all-mpnet-base-v2"):
    #store fields
    self.cache_dir = cache_dir
    self.data_df = data_df
    
    #create embedding function
    self.embedding_model = f'sentence-transformers/{embedding_model}'
    self.embedding_function = HuggingFaceEmbeddings(
        model_name=self.embedding_model,
        cache_folder=str(self.cache_dir)+'/huggingface_cache'
    )

    #Create a cross encoder for reranking
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
  def metadata_func(record:dict):
    """
    Create metadata for a vector

    Parameters
    ----------
    `dict`
        A record

    Returns
    -------
    `dict`
        A metadata record
    """
    keys = ['id','context','response']
    return {k:record[k] for k in keys if k in record}
  
  def ingest_df(self,df:pd.DataFrame):
    """
    ingest a dataframe into chromadb

    Parameters
    ---------
    df : `pd.DataFrame`
        pandas dataframe to ingest
    """
    #convert to records
    records = df.to_dict(orient='records')

    #loop through records create langchain document
    docs = []
    for record in records:
      raw_doc = Document(page_content=record['question'],
                          metadata = self.metadata_func(record))
      docs.append(raw_doc)

    #create a chroma from the documents using embedding function
    chroma = Chroma.from_documents(docs,self.embedding_function,
                                   persist_directory=str(self.chromadb_dir))

    #add documents to our vector store 
    self.vector_store._collection.add(
        embeddings = chroma.get()['embeddings'],
        metadatas = chroma.get()['metadatas'],
        documents = chroma.get()['documents'],
        ids = chroma.get()['ids']
    )

    #save!
    self.vector_store.persist()
  
  def rerank(self,query:str,retrieval:list):
    """
    Rerank the documents using CrossEncoder model

    Parameters
    ----------
    query : `str`
        the original question
    retrieval : `list`
        List of retrieved documents

    Returns
    -------
    `list`
        top three reranked documents
    """
    scores = self.reranker.predict(
      [(query,ret[0].page_content) for ret in retrieval]
    )
    return [retrieval[i] for i in np.argsort(scores)[::-1]][:3]
  
  def retrieve(self,query:str, k:int=4, key:str='response',as_prompt:bool=False, advanced:bool=False, 
               threshold:float=None):
    """
    Retrieve similar documents!

    Parameters
    ----------
    query : str
        The question
    k : int
        How many documents to retrieve
    key : `str`
        Which field in metadata to extract info from
    as_prompt : `bool`
        Convert document back into prompt format
    advanced : `bool`
        Use advanced reranking rather than naive
    threshold : `float`
        Minimum cosine sim for docs to be retrieved + used

    Returns
    -------
    `str` or `list`
        the document prompt or list
    """

    #if advanced, set k = 10
    if advanced:
      k = 10

    #perform retrieval
    retrieval = self.vector_store.similarity_search_with_relevance_scores(
        query,k
    )

    #rerank docs if asked
    if advanced:
      retrieval = self.rerank(query,retrieval)

    #filter out bad docs
    if threshold is not None:
        retrieval = [(r,i) for r,i in retrieval if i > threshold]

    #unpack results using key
    results = []
    for r in retrieval:
      results.append(r[0].metadata[key])

    #if not a prompt, return the list
    if not as_prompt:
      return results

    #return empty string if empty
    if len(results) == 0:
        return ""

    #else convert to prompt + return that.
    prompt = 'Use the following information to generate your answer:\n'
    for r in results:
      prompt += f'- {r}\n'
    return prompt