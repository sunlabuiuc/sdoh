#!/usr/bin/env python

"""`Convert`_ a spacy transformer model to a pytorch model.

.. _Convert: https://stackoverflow.com/questions/72414166/how-is-it-possible-to-use-the-spacytransformers-model-in-the-transfomers-pipel

"""

from spacy.lang.en import English
from spacy.pipeline import EntityRecognizer

# Load your saved spaCy model
import spacy
import os
import shutil

# Load your trained spaCy model
nlp = spacy.load('en_sdoh_roberta_cui')

checkpoint_model_name = 'roberta-base'
# Name of your new hf model
output_dir = f'sdoh-{checkpoint_model_name}'
os.makedirs(output_dir, exist_ok=True)

from transformers import PreTrainedTokenizerFast, RobertaTokenizerFast

# Convert spaCy tokenization to your model's standard tokenization (eg. wordpiece, bpe, etc.)

class CustomTokenizer(PreTrainedTokenizerFast):
    def __init__(self, spacy_tokenizer, backend_tokenizer, *args, **kwargs):
        super().__init__(tokenizer_object=backend_tokenizer, *args, **kwargs)
        self.spacy_tokenizer = spacy_tokenizer
        self._backend_tokenizer = backend_tokenizer

    def _tokenize(self, text):
        return [token.text for token in self.spacy_tokenizer(text)]

    def __getattr__(self, name):
        return getattr(self._backend_tokenizer, name)

    @property
    def backend_tokenizer(self):
        return self._backend_tokenizer

    def save_pretrained(self, save_directory, legacy_format=True, filename_prefix=None, push_to_hub=False, **kwargs):
        self._backend_tokenizer.save_pretrained(save_directory, legacy_format=legacy_format, filename_prefix=filename_prefix, push_to_hub=push_to_hub, **kwargs)


# Instantiate the custom tokenizer with the spaCy tokenizer and a backend tokenizer

spacy_tokenizer = nlp.tokenizer
backend_tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint_model_name)
custom_tokenizer = CustomTokenizer(spacy_tokenizer, backend_tokenizer)

# Save the tokenizer

custom_tokenizer.save_pretrained(output_dir)

# Save the model weights and configuration files
#nlp.config.to_disk(os.path.join(output_dir, 'config.json'))
import spacy
from transformers import AutoConfig
import json


# Get the label names from the named entity recognizer component
ner = nlp.get_pipe("ner")
label_names = ner.labels

# Create an AutoConfig object based on the spaCy model ... I finetuned a roberta-base model for NER, in my case ...
config = AutoConfig.from_pretrained(checkpoint_model_name, num_labels=len(label_names), id2label={i: label for i, label in enumerate(label_names)}, label2id={label: i for i, label in enumerate(label_names)})

# Save the configuration to disk in the Transformers-compatible format
config_dict = config.to_dict()
with open(os.path.join(output_dir, 'config.json'), 'w') as f:
    json.dump(config_dict, f)

nlp.vocab.to_disk(os.path.join(output_dir, 'vocab.txt'))
from transformers import RobertaForTokenClassification

# Create a Hugging Face model using the configuration object

hf_model = RobertaForTokenClassification.from_pretrained(checkpoint_model_name, config=config)

# Get the weights from the spaCy model and set the Hugging Face model weights
state_dict = {k.replace("roberta.", ""): v for k, v in nlp.get_pipe("transformer").model.transformer.named_parameters()}
state_dict["embeddings.position_ids"] = hf_model.roberta.embeddings.position_ids
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("pooler.")}
hf_model.roberta.load_state_dict(state_dict)

# Finally, save the Hugging Face model to disk

hf_model.save_pretrained(output_dir)
