import os, inflection, nltk
from random import shuffle
from tqdm.auto import tqdm

ENV = os.environ # global constant
ENV['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import spacy, spacy_transformers
from transformers import pipeline

### Model Loading ----------------------------------------------

SPACY_MODEL = 'en_core_web_trf'
if 'spacy_model' in os.environ:
    SPACY_MODEL = os.environ['spacy_model']

def get_spacy_model(model_name='auto'):
    env_name = 'SPACY_MODEL'
    
    if model_name == 'auto':
        if os.environ.get(env_name):
            model_name = os.environ.get(env_name)
            
        elif 'SPACY_MODEL' in globals():
            model_name = eval(env_name)

    model = spacy.load(model_name)

    print(f'spacy backend: {get_spacy_name(model)}')
            
    return model # specified spacy transformer

def get_spacy_name(model):
    return model.meta['lang']+'_'+model.meta['name']

### Typo Correction ----------------------------------------------

def load_spellcheck():
    if not 'fix_spelling' in globals():
        model = "oliverguhr/spelling-correction-english-base"
        fix_spelling = pipeline("text2text-generation", model)

    return fix_spelling

def correct_typos(texts):
    if not 'fix_spelling' in globals():
        fix_spelling = load_spellcheck()

    output = fix_spelling(text, max_length = 2048)
        
    return [x['generated_text'] for x in outputs]

### POS: NLTK --------------------------------------------------

def get_nltk_pos_descriptions(report=True):
    pos_tags = {
        "ADJ": "Adjective",
        "ADP": "Adposition (Pre- & Post-)",
        "ADV": "Adverb",
        "AUX": "Auxiliary Verb",
        "CONJ": "Conjunction",
        "CCONJ": "Coordinating Conjunction",
        "DET": "Determiner",
        "INTJ": "Interjection",
        "NOUN": "Noun",
        "NUM": "Numeral",
        "PART": "Particle",
        "PRON": "Pronoun",
        "PROPN": "Proper Noun",
        "PUNCT": "Punctuation",
        "SCONJ": "Subordinating Conjunction",
        "SYM": "Symbol",
        "VERB": "Verb",
        "X": "Other"
    } # dictionary of major pos_tags

    if not report:
        return pos_tags

    for pos_tag, description in pos_tags.items():
        print(f"{pos_tag}: {description}")

def extract_pos_nltk(sentence, kind = 'NOUN', exclude = []):
    pos_tag = {'nouns': 'NN', 'verbs': 'VB'}[kind]
    is_kind = lambda pos: pos_tag in pos[:2] 
    inputs = nltk.word_tokenize(sentence)
    return [word for (word, pos) in nltk.pos_tag(inputs) if is_kind(pos)
            and word not in exclude] 
    
### POS: SPACY --------------------------------------------------

def extract_pos_spacy(sentence, pos_tags, model='auto', exclude=False, 
                      lemmatize=False, plural_noun_lemmas=True):

    if model in ['auto', 'syntax_model']:
        if 'syntax_model' in globals():
            model = globals()['syntax_model']
        else: # load the model
            model = get_spacy_model()
    
    sentence = sentence.lower()
        
    doc = model(sentence)
    extracted_tokens = []
    
    def extract_lemma(token):
        lemma = token.lemma_
            
        if plural_noun_lemmas and token.tag_ == "NNS":
            lemma = inflection.pluralize(lemma)
        
        return lemma
        
    for token in doc:
        if not exclude:
            if token.pos_ in pos_tags:
                extracted_tokens.append(extract_lemma(token) if lemmatize else token.text)
                
        if exclude:
            if token.pos_ not in pos_tags:
                extracted_tokens.append(extract_lemma(token) if lemmatize else token.text)

    return extracted_tokens

def batch_extract_pos_spacy(sentences, pos_tags, model='auto', exclude=False, 
                            lemmatize=False, plural_nouns=False, **kwargs):

    if model in ['auto', 'syntax_model']:
        if 'syntax_model' in globals():
            model = globals()['syntax_model']
        else: # load the model
            model = get_spacy_model()
    
    sentences = [sentence.lower() for sentence in sentences]

    description = 'POS Extraction (over Sentences)'

    docs = [doc for doc in tqdm(model.pipe(sentences),
                                desc=description,
                                total=len(sentences))]

    def extract_lemma(token):
        lemma = token.lemma_
        if plural_nouns and token.tag_ == "NNS":
            lemma = inflection.pluralize(lemma)
        return lemma

    extracted_tokens_per_sentence = []

    for doc in docs:
        extracted_tokens = []
        for token in doc:
            if not exclude:
                if token.pos_ in pos_tags:
                    extracted_tokens.append(extract_lemma(token) if lemmatize else token.text)
            elif token.pos_ not in pos_tags:
                extracted_tokens.append(extract_lemma(token) if lemmatize else token.text)
        extracted_tokens_per_sentence.append(extracted_tokens)

    return extracted_tokens_per_sentence

### Perturbations --------------------------------------------------

extract_pos, batch_extract_pos = extract_pos_spacy, batch_extract_pos_spacy
    
    
