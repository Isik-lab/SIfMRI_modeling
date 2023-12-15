import spacy
import random
import string
import pandas as pd
from tqdm import tqdm


def strip_sentence(sentence):
    out = sentence.lower().rstrip('.').replace(',', '')
    return out.replace('[mask]', '[MASK]')


def shuffle_sentence(sentence):
    words = strip_sentence(sentence).split()  # Split sentence into words
    random.shuffle(words)  # Shuffle the words
    return ' '.join(words)  # Join the words back into a sentence


def mask_prep_phrases(text, filler='[MASK]', model_name='en_core_web_trf'):
    nlp = spacy.load(model_name)
    doc = nlp(text)

    masked_text = text
    spans_to_mask = []

    for token in doc:
        # Check for prepositions
        if token.dep_ == 'prep':
            # Extend the span to include the prepositional object and any modifiers
            span_start = token.idx
            if token.children:
                last_child = max(token.children, key=lambda x: x.i)
                span_end = last_child.idx + len(last_child.text)
                spans_to_mask.append((span_start, span_end))

    # Sort spans in reverse order to avoid indexing issues
    spans_to_mask.sort(key=lambda span: span[0], reverse=True)

    for start, end in spans_to_mask:
        masked_text = masked_text[:start] + filler + masked_text[end:]

    return strip_sentence(masked_text)


def mask_direct_objects(text, filler='[MASK]', model_name='en_core_web_trf'):
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


def mask_main_subjects(text, filler='[MASK]', model_name='en_core_web_trf'):
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


def mask_main_verb_phrase(text, filler='[MASK]', model_name='en_core_web_trf'):
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


def mask_adverbial_clauses(text, filler='[MASK]', model_name='en_core_web_trf'):
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


def mask_all_nouns(text, filler='[MASK]', model_name='en_core_web_trf'):
    nlp = spacy.load(model_name)
    doc = nlp(text)

    masked_text = text
    spans_to_mask = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            spans_to_mask.append((token.idx, token.idx + len(token.text)))

    # Sort spans in reverse order
    spans_to_mask.sort(key=lambda span: span[0], reverse=True)

    for start, end in spans_to_mask:
        masked_text = masked_text[:start] + filler + masked_text[end:]

    return strip_sentence(masked_text)


def mask_all_verbs(text, filler='[MASK]', model_name='en_core_web_trf'):
    nlp = spacy.load(model_name)
    doc = nlp(text)

    masked_text = text
    spans_to_mask = []

    for token in doc:
        # Check for main verbs and auxiliary verbs
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            spans_to_mask.append((token.idx, token.idx + len(token.text)))
        # Check for "to" in infinitives
        if token.text == "to" and token.dep_ == "aux":
            # Find the main verb of the infinitive
            main_verb = next((child for child in token.head.children if child.dep_ == "xcomp"), None)
            if main_verb:
                start = token.idx
                end = main_verb.idx + len(main_verb.text)
                spans_to_mask.append((start, end))

    # Sort spans in reverse order
    spans_to_mask.sort(key=lambda span: span[0], reverse=True)

    for start, end in spans_to_mask:
        masked_text = masked_text[:start] + filler + masked_text[end:]

    return strip_sentence(masked_text)


def load_grammarcheck(model_name="grammarly/coedit-xl-composite"):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def correct_grammar(text, prompt, tokenizer, model):
    input_ids = tokenizer(prompt + text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
