import os 
import pickle 

class lexicon:
    def __init__(self): 
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.lex_dict = None 
    
    def dictionary(self, idx:str):
        
        if idx == 'MASTER_v2022':   
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'MASTER', 'MASTER_v2022.pickle')           
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
        
        elif idx == "VADER_v2014_mod": 
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'VADER', 'VADER_v2014_mod.pickle')
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
                
        return self.lex_dict