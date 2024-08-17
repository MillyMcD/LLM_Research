import pandas as pd
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

class ChromaDB:
  """
  Vector Database via ChromaDB
  """
  def __init__(self,cache_dir,data_df=None):
    self.cache_dir = cache_dir
    self.data_df = data_df
    
    #create embedding function
    self.embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        cache_folder=str(self.cache_dir)+'/huggingface_cache'
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
    return {k:record[k] for k in keys}
  
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
  
  def retrieve(self,query,k=4,key:str='response'):
    """
    retrieve top k documents
    """
    retrieval = self.vector_store.similarity_search_with_relevance_scores(
        query,k)
  
    results = []
    for r in retrieval:
      print(r)
      results.append(r[0].metadata[key])
    return results