from init import * 

class lexicon:
    def __init__(self): 
        self.script_dir = SCRIPT_DIRECTORY
    
    def dictionary(self, idx:str):
        
        if idx == 'MASTER_v2022':   
            file_path = os.path.join(self.script_dir, 'lexicon_dictionary', 'MASTER', 'MASTER_v2022.pickle')           
            with open(file_path, 'rb') as handle: 
                lex_dict = pickle.load(handle)
        
        return lex_dict

lexicon = lexicon()
print(lexicon.dictionary("MASTER_v2022"))