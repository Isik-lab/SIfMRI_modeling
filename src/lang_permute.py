import os, inflection, nltk
import random
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

def pos_extraction(sentences, model, pos_tags, 
             lemmatize=False, shuffle=False,
                   exclude=False):
    
    def shuffle_words(sentence, seed=0):
        random.seed(seed) # set seed
        words = sentence.split(' ')
        random.shuffle(words)
        return ' '.join(words)
        
    if not isinstance(sentences, list):
        sentences = [sentences]
    
    extracts = batch_extract_pos(sentences, pos_tags, model,
                                              lemmatize=lemmatize, exclude=exclude)
    extracts = [' '.join(extract) for extract in extracts] 
    
    if shuffle:
        return [shuffle_words(words) for words in extracts]
    
    return extracts

def get_perturbation_data(perturb=None):
    out_conditions = {'orig_shuffled': {'pos': ['PUNC'], 'shuffle': True, 'exclude': True, 'lemmatize': False},
                      'orig_ordered': {'pos': ['PUNC'], 'shuffle': False, 'exclude': True, 'lemmatize': False},
                      'lemmas_shuffled': {'pos': ['PUNC'], 'shuffle': True, 'exclude': True, 'lemmatize': True},
                  'lemmas_ordered': {'pos': ['PUNC'], 'shuffle': False, 'exclude': True, 'lemmatize': True},
                  'excnv_shuffled': {'pos': ['NOUN', 'VERB'], 'shuffle': True, 'exclude': True, 'lemmatize': True},
                  'excnv_ordered': {'pos': ['NOUN', 'VERB'], 'shuffle': False, 'exclude': True, 'lemmatize': True},
                  'nv_shuffled': {'pos': ['NOUN', 'VERB'], 'shuffle': True, 'exclude': False, 'lemmatize': True},
                  'nv_ordered': {'pos': ['NOUN', 'VERB'], 'shuffle': False, 'exclude': False, 'lemmatize': True},
                  'verb_shuffled': {'pos': ['VERB'], 'shuffle': True, 'exclude': False, 'lemmatize': True},
                  'verb_ordered': {'pos': ['VERB'], 'shuffle': False, 'exclude': False, 'lemmatize': True},
                  'noun_shuffled': {'pos': ['NOUN'], 'shuffle': True, 'exclude': False, 'lemmatize': True},
                  'noun_ordered': {'pos': ['NOUN'], 'shuffle': False, 'exclude': False, 'lemmatize': True}
                  }
    if perturb is not None:
        return out_conditions[perturb]
    else:
        return list(out_conditions.keys())

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
    
    

