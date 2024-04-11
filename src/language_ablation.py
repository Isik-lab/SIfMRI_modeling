#
import spacy
import random
import pandas as pd
from tqdm import tqdm


def strip_sentence(sentence):
    out = sentence.lower().rstrip('.').replace(',', '')
    return out.replace('[mask]', '[MASK]')


class Masking: 
    def __init__(self, pos, mask_else, model_name='en_core_web_sm'):
        self.pos = pos 
        self.mask_else = mask_else 
        self.nlp = spacy.load(model_name)
        self.spacy_pos_lookup = {"nouns": ["NOUN", "PROPN"],
                                 "prepositions": ["ADP"],
                                 "verbs": ["VERB", "AUX"],
                                 "adjectives": ["ADJ"]}
        self.pos_search_list = self.spacy_pos_lookup(pos)
        self.filler = '[MASK]'
    
    def main(self, text): 
        # Load the SpaCy model
        doc = self.nlp(text)

        masked_text = text
        spans_to_mask = []
        for token in doc:
            if not self.mask_else: 
                bool_check = token.pos_ in self.pos_search_list
            else: 
                bool_check = token.pos_ not in self.pos_search_list

            if token.pos_ in self.pos_search_list:
                spans_to_mask.append((token.idx, token.idx + len(token.text)))

        # Sort spans in reverse order to avoid indexing issues during replacement
        spans_to_mask.sort(key=lambda span: span[0], reverse=True)

        for start, end in spans_to_mask:
            if not self.mask_else: 
                masked_text = masked_text[:start] + self.filler + masked_text[end:]
            else:
                masked_text = masked_text[:start] + self.filler * (end - start) + masked_text[end:]

        return strip_sentence(masked_text)


def shuffle_sentence(sentence):
    words = strip_sentence(sentence).split()  # Split sentence into words
    random.shuffle(words)  # Shuffle the words
    return ' '.join(words)  # Join the words back into a sentence


def mask_direct_objects(text, filler='[MASK]', model_name='en_core_web_sm'):
    nlp = spacy.load(model_name)
    doc = nlp(text)

    masked_text = text
    for token in doc:
        # Check if the token is a direct object
        if token.dep_ == 'dobj':
            start = token.idx
            end = start + len(token.text)
            masked_text = masked_text[:start] + filler + masked_text[end:]

    return strip_sentence(masked_text)


def mask_main_subjects(text, filler='[MASK]', model_name='en_core_web_sm'):
    nlp = spacy.load(model_name)
    doc = nlp(text)

    masked_text = text
    for sent in doc.sents:
        for token in sent:
            # Check for nominal subjects in the main clause
            if token.dep_ in ['nsubj', 'nsubjpass', 'csubj'] and token.head == sent.root:
                # Mask the subject span
                start = token.left_edge.idx
                end = token.right_edge.idx + len(token.right_edge.text)
                masked_text = masked_text[:start] + filler + masked_text[end:]
                # Update the doc to reflect the changes
                doc = nlp(masked_text)

    return strip_sentence(masked_text)


def mask_main_verb_phrase(text, filler='[MASK]', model_name='en_core_web_sm'):
    nlp = spacy.load(model_name)
    doc = nlp(text)

    masked_text = text
    for sent in doc.sents:
        root = sent.root  # The root of the sentence is typically the main verb
        # Find the span of the main verb and its immediate dependents
        start = root.idx
        end = root.idx + len(root.text)
        for child in root.children:
            if child.dep_ in ['aux', 'auxpass', 'advmod', 'neg']:
                # Extend the span to include auxiliaries or adverbs
                if child.idx < start:
                    start = child.idx
                elif child.idx + len(child.text) > end:
                    end = child.idx + len(child.text)

        # Replace the main verb phrase with [MASK]
        masked_text = masked_text[:start] + filler + masked_text[end:]

    return strip_sentence(masked_text)


def mask_adverbial_clauses(text, filler='[MASK]', model_name='en_core_web_sm'):
    nlp = spacy.load(model_name)
    doc = nlp(text)

    masked_text = text
    spans_to_mask = []

    for token in doc:
        # Check for common subordinating conjunctions that introduce adverbial clauses
        if token.dep_ == 'mark' and token.head.pos_ == 'VERB':
            # Find the span of the adverbial clause
            span_start = token.head.left_edge.i
            span_end = token.head.right_edge.i
            spans_to_mask.append((span_start, span_end))

    # Sort spans in reverse order to avoid indexing issues
    spans_to_mask.sort(key=lambda span: span[0], reverse=True)

    for span_start, span_end in spans_to_mask:
        start = doc[span_start].idx
        end = doc[span_end].idx + len(doc[span_end].text)
        masked_text = masked_text[:start] + filler + masked_text[end:]

    return strip_sentence(masked_text)


def load_grammarcheck(model_name="grammarly/coedit-xl-composite"):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def correct_grammar(prompt, text, tokenizer, model):
    input_ids = tokenizer(prompt + text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
