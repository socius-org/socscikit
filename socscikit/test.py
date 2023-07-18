import CompliSent

lexicon = CompliSent.lexicon()
VADER = lexicon.load_dict('VADER_v2014_mod')
lexicon.overview(VADER)