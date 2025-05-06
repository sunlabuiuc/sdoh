#!/usr/bin/env python

from pathlib import Path
import spacy
import pandas as pd


def main():
    out_file = Path(f'sdoh-ner-cmp/{spacy.__version__}')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load('en_sdoh_bow')
    paths = filter(
        lambda p: p.suffix == '.csv',
        Path('corpus/feature_resource').iterdir())
    print(f'writing results to {out_file}')
    with open(out_file, 'w') as f:
        for p in paths:
            df: pd.DataFrame = pd.read_csv(p)
            for text in df['text']:
                try:
                    doc = nlp(text)
                    ents = str(doc.ents)
                except Exception as e:
                    ents = f'error<<{e}>>'
                print(f'{text}:{ents}', file=f)


if (__name__ == '__main__'):
    main()
