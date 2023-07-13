import pickle 
import os 

class lexicon:
    def __init__(self): 
        self.directory = os.getcwd()
    
    def dictionary(self, idx:str):
        if idx == 'MASTER_v2022':              
            with open('CompliSent/lexicon_dictionary/MASTER/MASTER_v2022.pickle', 'rb') as handle: 
                lex_dict = pickle.load(handle)
        
        return lex_dict