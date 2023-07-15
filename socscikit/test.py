from socscikit import CompliSent

lexicon = CompliSent.lexicon()
MASTER = lexicon.load_dict('MASTER_v2022')
lexicon.overview(MASTER)