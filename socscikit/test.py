from socscikit import CompliSent

# lexicon = CompliSent.lexicon()
# VADER = lexicon.load_dict('AFINN_v2015')
# lexicon.overview(VADER)

import gradio as gr
import os
import spacy
from spacy import displacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()

def text_analysis(text):
    

    doc = nlp(text)
    html = displacy.render(doc, style="dep", page=True)
    html = (
        "<div style='max-width:200%; max-height:360px; overflow:auto'>"
        + html
        + "</div>"
    )
    
    scores = sid.polarity_scores(text)
    del scores["compound"]
    
    pos_tokens = []

    for token in doc:
        pos_tokens.extend([(token.text, token.pos_), (" ", None)])

    return pos_tokens, scores, html

demo = gr.Interface(
    text_analysis,
    gr.Textbox(placeholder="Enter sentence here..."),
    ["highlight", "label", "html"],
    examples=[
        ["Apple is about to buy UK start-up at $1 billion."],
        ["It looks like we are having a really good day!!"],
    ],
)

demo.launch()