from socscikit import CompliSent

lexicon = CompliSent.lexicon()
VADER = lexicon.load_dict('AFINN_v2015')
lexicon.overview(VADER)