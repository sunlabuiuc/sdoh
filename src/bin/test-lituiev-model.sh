#!/usr/bin/env python

import spacy

text = 'Owner operator food truck selling gourmet hamburgers around town'
expected_label = 'food'
nlp = spacy.load('en_sdoh_bow')
doc = nlp(text)
label = doc.ents[0].orth_
if label != expected_label:
    raise ValueError(
        f'expected singleton entity {expected_label} but got: {label}')
