import time
import json
import pandas as pd
import ollama

from tqdm import tqdm
from pathlib import Path
from src.vectordb import ChromaDB

class QuestionAnswering:
    """
    Class to ask a language model questions + store them. Requires a dataframe containing
    'question' and 'response'

    Parameters
    ----------
    model : `str`
        name of the LLM supported by Ollama.
    """
    def __init__(self,model:str='llama3'):
        self.model = model

    def process_dataset(self,df:pd.DataFrame):
        """
        Convert dataset to records i.e. our dataframe becomes a list of dictionaries, where
        the dictionary contains the question and answer pair.
        """
        self.records = df.to_dict(orient='records') #splits into pairs to loop through

    def system_prompt(self):
        """
        Create a system prompt

        Returns
        -------
        `str`
            The system prompt
        """
        return '''You are to be a human-like, compassionate, friendly and polite
                chatbot assistant for a cyber security firm. You will be asked customer support
                questions and it is your job to answer those questions. You aim to answer all
                queries, and if you are unsure you will ask the customer to hold while they
                are transferred to a human agent.'''

    def ask_question(self,record:dict,vector_db:ChromaDB=None,k:int=1,advanced:bool=False):
        """
        Ask a question using the record

        Parameters
        ----------
        record : `dict`
            Dictionary containing the question and answer
        vector_db : `ChromaDB`
            A ChromaDB object
        k : `int`
            Number of documents to retrieve
        advanced : `bool`
            Use naive RAG (False) or advanced reranking (True)

        Returns
        -------
        `dict`
            A record!
        """

        #this copies record into a new dictionary to avoid overwriting original
        rec = {ki:v for ki,v in record.items()} 

        #create the prompts. System and user
        system = self.system_prompt()
        prompt = record['question']

        #retrieve document
        if vector_db is not None:
          prompt += ' '
          prompt += vector_db.retrieve(record['question'],k=k,as_prompt=True,
                                        advanced=advanced)

        #call LLM. create a timer too
        start = time.time()
        llm_response = ollama.generate(model = self.model, system = system, prompt = prompt,
                                       options = {'temperature':0.0,
                                                  'num_predict':1000})
        end = time.time()
        
        record['llm_response'] = llm_response['response']
        record['time'] = end - start
        record['tps'] = len(record['llm_response'].split()) / record['time']
        return record

    def ask_all_questions(self,save_path:str|Path,vector_db:ChromaDB=None,k:int=1,advanced:bool=False):
        """
        Ask all questions in the dataset

        Parameters
        ----------
        save_path: `str`
            Location to save your results
        vector_db : `ChromaDB`
            A ChromaDB object
        k : `int`
            Number of documents to retrieve
        advanced : `bool`
            Use naive RAG (False) or advanced reranking (True)

        Returns
        -------
        `pd.DataFrame`
            The results!
        """
        #create folder
        enriched_records = []
        folder = Path(save_path) / self.model
        folder.mkdir(exist_ok=True,parents=True)

        #loop through records and enrich
        for i,q in enumerate(tqdm(self.records)):
            id = q['id']
            resp = self.ask_question(record=q,vector_db=vector_db,k=k,
                                    advanced = advanced)
            resp['model'] = self.model
            with open(folder/f'{id}.json','w') as f:
              json.dump(resp,f)
            enriched_records.append(resp)
        #save!
        save_name = folder/'all_questions.csv'
        resp_df = pd.DataFrame(enriched_records)
        resp_df.to_csv(save_name,index=False)
        return resp_df