import ollama
import pandas as pd
from src.vectordb import ChromaDB

def system_prompt_generic():
    s = "You are a chatbot assistant for the Cybersecurity company METCLOUD. You should be friendly but very professional "
    s += "You should refer to METCLOUD as 'us' and 'we'. "
    s += "Read the question carefully and assess if you can reasonably or accurately answer it. If you cannot, apologise and justify why. "
    s += "You should be especially dilligent about METCLOUD SPECIFIC questions such as location, staff or services... but do not explain this in your answer. "
    s += "Under no circumstances should you make up ficticious information relating to the question. If you detect that you cannot answer this "
    s += "question, respond by offering to hand off to a METCLOUD 'superassistant' and asking the user if they would like to do that. "
    s += 'Remember, do not attempt to embellish METCLOUDs capabilities and do not make us liable to damages. '
    return s

def system_prompt_with_context():
    s = system_prompt_generic()
    s += "To help you answer your question, I have provided you additional context that you should use to base your answer against. Make sure you "
    s += "tailor this information towards the original question. The same rules apply; do NOT make up ficticious information"
    return s

def system_prompt_handoff():
    s = "You are a chatbot assistant for the Cybersecurity company METCLOUD. You should be friendly but very professional. "
    s += "You should refer to METCLOUD as 'us' and 'we'. "
    s += "In this circumstance, the USER has asked a METCLOUD specific question that is outside of our knowledge base. "
    s += "We should therefore respond by offering to hand off to a METCLOUD 'superassistant' who can help them with their query. "
    return s

def system_prompt_nonspecific():
    s = system_prompt_generic()
    s = "You are a chatbot assistant for the Cybersecurity company METCLOUD. You should be friendly but very professional. "
    s += "You should refer to METCLOUD as 'us' and 'we'. "
    s += "Read the question carefully and assess if the question is the sort of thing a Cyber security professional could be expected to be asked. "
    s += "If the answer that question is YES, answer to the best of your abilities but explain that your understanding is only approximate. If the "
    s += "answer is NO, apologise and say that you cannot help, and explain what you _can_ help with i.e METCLOUD + Cyber help. "
    s += "Remember, do not attempt to embellish METCLOUDs capabilities and do not make us liable to damages. "
    return s

def is_metcloud_specific(question:str,model):
    s = '# YOUR ROLE\n'
    s += 'You are a specialised text analysis model designed to detect questions specifically related to METCLOUD, a company '
    s += 'offering cybersecurity services. Your task is to analyse the input text and deetermine if it contains a question about '
    s += 'METCLOUD or its services\n'
    s += '# INSTRUCTIONS\n'
    s += '1. Carefully read the text and analyse it\n'
    s += '2. Determine if the text contains a query about METCLOUD or its services\n'
    s += '3. Respond ONLY with "True" if the text contains a specific question, or "False" if it does not\n\n'
    s += 'Consider a question METCLOUD-specific if it:\n'
    s += ' - Directly mentions METCLOUD by name\n'
    s += ' - Asks about cybersecurity services\n'
    s += ' - Inquires about account details, pricing or support relating to METCLOUD\n\n'
    s += 'DO NOT RESPOND "True" for:\n'
    s += '- General questions about cyber security \n'
    s += '- Questions about other companies not relating to METCLOUD\n\n'
    s += '# OUTPUT INSTRUCTIONS\n'
    s += 'Respond with only a single string "True" or "False". No yapping! Do not act as an assistant.'

    response = ollama.generate(model = model,system=s,prompt = f'Is this question METCLOUD specific? : {question}',options = {'temperature':0.0})
    judgement =  response['response'].lower()
    return True if 'true' in judgement else False

class Pipeline:
    def __init__(self,llm_model:str,emb_model:str,corpus:pd.DataFrame,cache:str = '/data/hand_off_pipeline'):
        
        self.llm_model = llm_model
        self.emb_model = emb_model
        self.corpus    = corpus
        self.cache     = cache
                        
        self.chroma_db = ChromaDB(
            self.cache,
            self.corpus,
            embedding_model = self.emb_model
        )

    def ask_question(self,question:str,threshold:float=0.5,advanced:bool=False):
        context = self.chroma_db.retrieve(question,threshold=threshold,as_prompt=True,k=1,advanced=advanced)

        if len(context) == 0:
            if is_metcloud_specific(question,model = self.llm_model):
                print('METCLOUD Specific question asked, outside of our context!')
                response = ollama.generate(model = self.llm_model, system = system_prompt_handoff(), 
                                           prompt = 'Use the system message instructions to respond')
                return response['response']
            else:
                print('Generic question asked, outside of our context!')
                response = ollama.generate(model = self.llm_model, system = system_prompt_nonspecific(), 
                                           prompt = question)
                return response['response']
        else:
            print('Info retrieved!')
            user_prompt = question + '\n' + context
            response = ollama.generate(model = self.llm_model, system = system_prompt_with_context(), 
                                       prompt = user_prompt)
            return response['response']
        

