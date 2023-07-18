from socscikit import CompliSent

lexicon = CompliSent.lexicon()
MASTER = lexicon.load_dict('VADER_v2014_mod')
lexicon.overview(MASTER)