import json
import re
import ollama
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from src.semscore import sem_score

class Verifier:
  """
  Get a LLM to mark if generated answer is consistent with the ground-truth.

  Parameters
  ----------
  model : `str`
      Name of LLM, compatible with Ollama
  embedding_model : `str`
      name of sentence-transformer model. For SemScore
  cache_dir : `str`
      Place to store weights
  """
  def __init__(self,model:str='phi3',embedding_model:str = "all-mpnet-base-v2",cache_dir:str = '/data/cache'):
    #llm model
    self.model=model

    #embedding function for Semantic Score
    embedding_model = f'sentence-transformers/{embedding_model}'
    self.emb_func = HuggingFaceEmbeddings(
        model_name = embedding_model,
        cache_folder = cache_dir
    )

  def system_prompt(self,gen_response:str,ground_truth_answer:str):
    """
    system prompt for verification

    Parameters
    ----------
    gen_response : `str`
        LLM generated response
    ground_truth_answer : `str`
        Ground truth answer

    Return
    ------
    `str`
    """
    format = {'consistent':'(either "True" or "False")',
               'justification':'(description why the samples are consistent)'}
      
    p  = '# YOUR ROLE\n'
    p += 'You are a question and answering validation capability. '
    p += 'You can accurately compare two potential pieces of text for similarity and consistency.'

    p += '\n\n# YOUR TASK\n'
    p += f'Your task is to assess / judge if information A : {gen_response}\n'
    p += f'IS CONSISTENT with information B: {ground_truth_answer}\n'
    p += 'Information B should be treated as the TRUTH even if you disagree with its content. '
    p += 'If the text samples contain similar and non-conflicting information, '
    p += 'then then you should judge them as consistent. '

    p += '\n\n# OUTPUT INSTRUCTIONS\n'
    p += 'Return your judgement as a JSON compatible dictionary. An example of '
    p += 'this format is:\n\n'
    p += json.dumps(format)
    p += '\n\nYour output should only contain the "consistent" and "justification". '
    p += 'Do not act as an assistant '
    p += 'and do not yap. Make sure your output is valid JSON.'
    return p
  
  def extract_and_parse(self,text:str,required_fields:dict):
    """
    Extract/parse response into a dictionary. Makes sure format
    is respected

    Parameters
    ----------
    text : `str`
        LLM generated response
    required_fields : `dict`
        fields for dictionary to contain

    Returns
    -------
    `dict`
        The response
    """
    text = text.replace("(","")
    text = text.replace(")","")

    #use regex to find pattern matching dict
    matches = re.findall(r'\{(.*?)\}', text)
    for match in matches:
      #reconstruct dictionary + impose fields
      data = eval('{' + match + '}')
      data = {k:v for k,v in data.items() if k in required_fields}
      return data
    #return stock otherwise
    return {'consistent':False,'justification':'missed'}

  def judge_llm(self,gen_response:str,ground_truth_answer:str,
                       temperature:float=0.0,seed:int=1000):
    """
    Judge if response is consistent

    Parameters
    ----------
    gen_response : `str`
        LLM generated response
    ground_truth_answer : `str`
        Ground truth answer
    temperature : `float`
        the llm temperature
    seed : `int`
        the seed
    """
    #get system prompt
    system = self.system_prompt(gen_response,ground_truth_answer)
    prompt = 'Compare the information'

    #call ollama
    response = ollama.generate(model=self.model,system=system,prompt=prompt,
                               options={'temperature':temperature,'seed':seed})

    #sanitise output
    fields = ['consistent','justification']
    response = self.extract_and_parse(response['response'],fields)

    return response

  def judge_sem_score(self,gen_response:str,ground_truth_answer:str):
      """
      Get semantic score

      Parameters
      ----------
      gen_response : `str`
        LLM generated response
      ground_truth_answer : `str`
        Ground truth answer

      Returns
      -------
      `float`
          semscore
      """
      return sem_score(
          gen_response,
          ground_truth_answer,
          self.emb_func)

  def judge_all_questions(self,df:pd.DataFrame,model:str,save_dir:str):
    """
    Judge all question responses

    Parameters
    ----------
    df : `pandas.DataFrame`
        the dataset containing generated and ground truth strings
    model : `str`
        Name of the model being assessed
    save_dir : `str`
        Where to save the results
    """

    #split into records
    records = df.to_dict(orient='records')

    #loop through records
    marked = []
    for rec in tqdm(records):
      gt = str(rec['response'])
      pr = str(rec['llm_response'])
      id = rec['id']

      #add llm judge response
      resp = self.judge_llm(pr,gt)
      
      nrec = {'id':id}
      nrec.update(resp)

      #add sem score
      if self.emb_func is not None:
          nrec['sem_score'] = self.judge_sem_score(gt,pr)

      marked.append(nrec)

    #combine results
    marked_df = pd.DataFrame(marked)
    marked_df['mean_time'] =df['time'].mean()
    marked_df['mean_tps'] =df['tps'].mean()

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True,parents=True)

    #group up the results
    marked_df['accuracy'] = marked_df['consistent'].value_counts()['True'] / len(marked_df)
    if self.emb_func is not None:
        marked_df['sem_acc'] = sum(marked_df['sem_score']>0.7)/len(marked_df)
    marked_df.to_csv(save_dir/f'{model}.csv',index=False)

    

    
    

  
  