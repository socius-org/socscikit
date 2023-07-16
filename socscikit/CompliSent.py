import os 
import pickle 
from socscikit.utils import CS 

class lexicon:
    def __init__(self): 
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.lex_dict = None 
    
    def load_dict(self, idx:str):
        
        if idx == 'MASTER_v2022':   
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'MASTER', 'MASTER_v2022.pickle')           
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
        
        elif idx == "VADER_v2014_mod": 
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'VADER', 'VADER_v2014_mod.pickle')
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
                
        elif idx == "AFINN_v2015": 
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'AFINN', 'AFINN_v2015.pickle')
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
        
        elif idx == "Aigents+_v2022": 
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'Aigents', 'Aigents+_v2022.pickle')
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
                
        return self.lex_dict
    
    def overview(self, dictionary:dict): 
        #dict:str = Consider if users simply come up with the lex_dict_idx
        return CS().summarise_lex_dict(dictionary)