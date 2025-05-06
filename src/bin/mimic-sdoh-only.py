#!/usr/bin/env python

"""Render the MIMIC annotated corpus and annotations.

"""

from pathlib import Path
import pandas as pd
from zensols.rend import ApplicationFactory


def main():
    path = Path('corpus/feature_resource/SDOH_MIMICIII_physio_release.csv')
    print(f'corpus file: {path}')
    df: pd.DataFrame = pd.read_csv(path)
    dfd = df.drop(columns=[
        'provider_type', 'patient_id', 'note_id', 'sentence_index', 'text'])
    dfd = dfd[dfd.gt(0).any(axis=1)]
    df = df.loc[dfd.index]
    print(df)
    ApplicationFactory.render(df)


if (__name__ == '__main__'):
    main()
