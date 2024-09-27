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
  Get a LLM to mark if generated answer is consistent with the ground-truth
  """
  def __init__(self,model:str='phi3',embedding_model = "all-mpnet-base-v2",cache_dir:str = '/data/cache'):
    self.model=model

    embedding_model = f'sentence-transformers/{embedding_model}'
    self.emb_func = HuggingFaceEmbeddings(
        model_name = embedding_model,
        cache_folder = cache_dir
    )

  def system_prompt(self,gen_response:str,ground_truth_answer:str):
    """
    system prompt for verification
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
    """
    text = text.replace("(","")
    text = text.replace(")","")
    
    matches = re.findall(r'\{(.*?)\}', text)
    for match in matches:
      data = eval('{' + match + '}')
      data = {k:v for k,v in data.items() if k in required_fields}
      return data
    return {'consistent':False,'justification':'missed'}

  def judge_llm(self,gen_response:str,ground_truth_answer:str,
                       temperature:float=0.0,seed:int=1000):
    """
    Judge if response is consistent
    """
    system = self.system_prompt(gen_response,ground_truth_answer)
    prompt = 'Compare the information'

    response = ollama.generate(model=self.model,system=system,prompt=prompt,
                               options={'temperature':temperature,'seed':seed})

    fields = ['consistent','justification']
    response = self.extract_and_parse(response['response'],fields)

    return response

  def judge_sem_score(self,gen_response,ground_truth_answer):
      return sem_score(
          gen_response,
          ground_truth_answer,
          self.emb_func)

  def judge_all_questions(self,df,model,save_dir):
    """
    Judge all question responses
    """
    records = df.to_dict(orient='records')

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

    marked_df = pd.DataFrame(marked)
    marked_df['mean_time'] =df['time'].mean()
    marked_df['mean_tps'] =df['tps'].mean()

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True,parents=True)
    
    marked_df['accuracy'] = marked_df['consistent'].value_counts()['True'] / len(marked_df)
    if self.emb_func is not None:
        marked_df['sem_acc'] = sum(marked_df['sem_score']>0.7)/len(marked_df)
    marked_df.to_csv(save_dir/f'{model}.csv',index=False)

    

    
    

  
  