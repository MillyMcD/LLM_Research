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
    s += "Read the question carefully and assess if the question relates to the field of Cyber security and cloud services. "
    s += "If YES, answer to the best of your abilities, but explain your understanding is only approximate for this question. "
    s += "If NO, apologise and explain that this question is unrelated to your expertise. Do not try to answer the question. "
    s += "In summary, only attempt to answer Cyber Security questions! Importantly, do NOT embellish METCLOUDs capabilities. "
    s += "We do not want to be liable for damages!"
    return s

def system_prompt_cyber():
    s = "You are a chatbot assistant for the Cybersecurity company METCLOUD. You should be friendly but very professional "
    s += "You should refer to METCLOUD as 'us' and 'we'. "
    s += "You have generic cyber security expertise, but you only really understand METCLOUD to a high degree of expertise "
    s += "Answer any cyber related questions, but you should ALWAYS explain that this understanding is only approximate "
    s += "because you are built for METCLOUD services. Importantly, do NOT embellish METCLOUDs capabilities. "
    s += "We do not want to be liable for damages! Remember, we are METCLOUD and only have knowledge on how to do METCLOUD things. "
    s += "We should only offer rough advice for helping with their cyber issues, and should direct people to more appropriate sources wherever possible"
    return s

def system_prompt_general():
    s = "You are a chatbot assistant for the Cybersecurity company METCLOUD. You should be friendly but very professional. "
    s += "You should refer to METCLOUD as 'us' and 'we'. "
    s += "In this circumstance, the USER has asked a general question that is outside of our knowledge base and is UNRELATED to our expertise. "
    s += "You should apologise, and explain that we cannot answer this question. Offer to answer a different more appropriate question. "
    return s
    
def is_metcloud_specific(question:str,model):
    s = '# YOUR ROLE\n'
    s += 'You are a specialised text analysis model designed to detect questions specifically related to METCLOUD, a company '
    s += 'offering cybersecurity services. Your task is to analyse the input text and determine if it contains a question about '
    s += 'METCLOUD or its services\n\n'
    s += '# INSTRUCTIONS\n'
    s += '1. Carefully read the text\n'
    s += '2. Determine if the text contains a query about METCLOUD or its services\n'
    s += '3. Respond ONLY with "True" if the text contains METCLOUD, or "False" if it does not\n\n'
    s += 'Consider a question METCLOUD-specific if it matches any of the following criteria:\n'
    s += ' - the word METCLOUD appears in the question\n'
    s += ' - Asks about cybersecurity services\n'
    s += ' - Inquires about account details, pricing or support relating to METCLOUD\n\n'
    s += 'DO NOT RESPOND "True" for:\n'
    s += '- General questions about cyber security \n'
    s += '- Questions about other companies not relating to METCLOUD\n\n'
    s += '# OUTPUT INSTRUCTIONS\n'
    s += 'Respond with only a single string "True" or "False". No yapping! Do not act as an assistant\n'
    s += '# SPECIAL INSTRUCTIONS\n'
    s += 'Read the question token by token. If you see "metcloud" or "METCLOUD", then return "True"'
    
    response = ollama.generate(model = model,system=s,prompt = f'Does this question mention METCLOUD? : {question}',options = {'temperature':0.0})
    judgement =  response['response'].lower()
    return True if 'true' in judgement else False


def is_cyber(question:str,model):
    s = '# YOUR ROLE\n'
    s += 'You are a specialised text analysis model designed to detect questions specifically related to CYBER SECURITY, cyber defence and so forth. '
    s += 'Your task is to analyse the input text and determine if it contains a question about cyber related topics and activities\n\n'
    s += '# INSTRUCTIONS\n'
    s += '1. Carefully read the text\n'
    s += '2. Determine if the text contains a query about Cyber Security, Cyber defence or related.\n'
    s += '3. Respond ONLY with "True" if the text does relate to cyber or "False" if it does not\n\n'
    s += 'Consider a question related to cyber if it matches any of the following criteria:\n'
    s += ' - Technical networking questions are asked\n'
    s += ' - Asks about cybersecurity or cyber defence\n'
    s += ' - Asks about APT, TTPs or similar\n'
    s += ' - talks about network issues, hacking and so forth\n'
    s += 'DO NOT RESPOND "True" for:\n'
    s += '- General questions about unrelated topics such as pop culture'
    s += '# OUTPUT INSTRUCTIONS\n'
    s += 'Respond with only a single string "True" or "False". No yapping! Do not act as an assistant\n'
    
    response = ollama.generate(model = model,system=s,prompt = f'Does this question relate to cyber or cyber security? : {question}',options = {'temperature':0.0})
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
                return response['response'], 'metcloud_specific'
            else:
                if is_cyber(question,model = self.llm_model):
                    print('Cyber question!')
                    response = ollama.generate(model = self.llm_model,
                                               system = system_prompt_cyber(),
                                               prompt = question)
                    return response['response'], 'generic_cyber'
                else:
                    print('Generic Question!')
                    response = ollama.generate(model = self.llm_model,
                                               system = system_prompt_general(),
                                               prompt = f'use the system message instructions to explain why you cannot answer this question: {question}')
                    return response['response'], 'generic_external'
        else:
            print('Info retrieved!')
            user_prompt = question + '\n' + context
            response = ollama.generate(model = self.llm_model, system = system_prompt_with_context(), 
                                       prompt = user_prompt)
            return response['response'], 'retrieval'
        

