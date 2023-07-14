import os 
import pickle 

class lexicon:
    def __init__(self): 
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def dictionary(self, idx:str):
        
        if idx == 'MASTER_v2022':   
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'MASTER', 'MASTER_v2022.pickle')           
            with open(file_path, 'rb') as handle: 
                lex_dict = pickle.load(handle)
        
        elif idx == "VADER": 
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'VADER', 'VADER.pickle')
            with open(file_path, 'rb') as handle: 
                lex_dict = pickle.load(handle)
                
        return lex_dict