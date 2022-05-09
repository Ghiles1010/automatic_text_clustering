import spacy
import pandas as pd
import re

def load_spacy(lang="en_core_web_sm", exclude=None):

    if exclude is None:
        exclude = ['parser', 'tok2vec', 'lemmatizer', 'tagger', 'attribute_ruler']

    return spacy.load(lang, exclude=exclude)


def clean(texts, nlp):
    docs= nlp.pipe(texts)
    texts = []
    for doc in docs:
        tokens = [token.text for token in doc if not token.ent_type_ and not token.is_stop and not token.is_punct and not token.is_space]
        texts.append(" ".join(tokens))
    return texts


def preprocess(texts, person_re="#.+?#", nlp=None):

    if nlp is None:
        nlp = load_spacy()

    sr = pd.Series(texts)

    rep = lambda x : re.sub(person_re, "", x)

    sr = sr.apply(rep)
    sr = sr.apply(lambda x : x.lower())
    sr = clean(sr, nlp)

    return sr