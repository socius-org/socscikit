import pickle 


with open('CompliSent/lexicon_dictionary/MASTER/MASTER_v2022.pickle', 'rb') as handle: 
    MASTER_v2022 = pickle.load(handle)

print(MASTER_v2022)