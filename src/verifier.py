import json
import re
import ollama

class Verifier:
  """
  Get a LLM to mark if generated answer is consistent with the ground-truth
  """
  def __init__(self,model:str='phi3'):
    self.model=model

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
    p += f'Your task is to assess the judge if information A : {gen_response}\n'
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
  
  def extract_and_parse(text:str,required_fields:dict):
    """
    Extract/parse response into a dictionary. Makes sure format
    is respected
    """
    matches = re.findall(r'\{(.*?)\}', text)
    for match in matches:
      data = eval('{' + match + '}')
      data = {k:v for k,v in data.items() if k in required_fields}
      return data

  def judge(self,gen_response:str,ground_truth_answer:str,
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
    

  
  