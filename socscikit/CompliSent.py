import os 
import pickle 
from socscikit.utils import CS 

class lexicon:
    def __init__(self): 
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.lex_dict = None 
    
    def load_dict(self, idx:str):
        """_summary_

        Args:
            idx (str): _description_

        Returns:
            _type_: _description_
        """
        if idx == 'MASTER_v2022':   
            file_path = os.path.join(self.script_dir, 'dict_arXiv', 'MASTER', 'MASTER_v2022.pickle')           
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
        
        elif idx == "VADER_v2014_mod": 
            file_path = os.path.join(self.script_dir, 'dict_arXiv', 'VADER', 'VADER_v2014_mod.pickle')
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
                
        elif idx == "AFINN_v2015": 
            file_path = os.path.join(self.script_dir, 'dict_arXiv', 'AFINN', 'AFINN_v2015.pickle')
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
        
        elif idx == "Aigents+_v2022": 
            file_path = os.path.join(self.script_dir, 'dict_arXiv', 'Aigents', 'Aigents+_v2022.pickle')
            with open(file_path, 'rb') as handle: 
                self.lex_dict = pickle.load(handle)
                
        return self.lex_dict
    
    def load_origin(self, idx:str): 
        """_summary_

        Args:
            idx (str): _description_

        Returns:
            _type_: _description_
        """
        import pandas as pd 
        
        if idx == 'MASTER_v2022':   
            file_path = os.path.join(self.script_dir, 'dict_arXiv', 'MASTER', 'MASTER_v2022.csv')           
            self.origin_df = pd.read_csv(file_path)
        
        elif idx == "VADER_v2014_mod": 
            file_path = os.path.join(self.script_dir, 'dict_arXiv', 'VADER', 'VADER_v2014_mod.csv')
            self.origin_df = pd.read_csv(file_path)
                
        elif idx == "AFINN_v2015": 
            file_path = os.path.join(self.script_dir, 'dict_arXiv', 'AFINN', 'AFINN_v2015.csv')
            self.origin_df = pd.read_csv(file_path)
        
        elif idx == "Aigents+_v2022": 
            file_path = os.path.join(self.script_dir, 'dict_arXiv', 'Aigents', 'Aigents+_v2022.csv')
            self.origin_df = pd.read_csv(file_path)
        
        return self.origin_df
    
    def overview(self, dictionary:dict=None): 
        #dict:str = Consider if users simply come up with the lex_dict_idx
        if dict is None: 
            raise ValueError
        else:
            return CS().summarise_lex_dict(dictionary)