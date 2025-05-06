"""SDoH Corpus.

"""
__author__ = 'Paul Landes'

from typing import List, Iterable, Tuple, Sequence
from dataclasses import dataclass, field
import logging
from pathlib import Path
import pandas as pd
from zensols.util import APIError
from zensols.persist import persisted
from zensols.dataframe import ResourceFeatureDataframeStash

logger = logging.getLogger(__name__)


@dataclass
class SdohCorpusStash(ResourceFeatureDataframeStash):
    mimic_corpus_file: str = field()
    mimic_corpus_columns: str = field()
    none_label: str = field()
    synthetic_files: Sequence[str] = field()

    @persisted('_parent_path')
    def _get_parent_path(self) -> Path:
        self.installer.install()
        return self.installer.get_singleton_path()

    def _get_mimic_dataframe(self) -> pd.DataFrame:
        def map_row_labels(row: pd.Series) -> Iterable[str]:
            cols: Tuple[str, ...] = tuple(sorted(filter(
                lambda c: row[c] > 0, self.mimic_corpus_columns)))
            if len(cols) > 0:
                return cols
            return (self.none_label,)

        par_dir: Path = self._get_parent_path()
        csv_file: Path = par_dir / self.mimic_corpus_file
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'loading {csv_file}')
        df: pd.DataFrame = pd.read_csv(csv_file)
        # collapse labels
        to_drop: List[str] = []
        col: str
        for col in df.columns:
            mcol: str
            for mcol in self.mimic_corpus_columns:
                if col.startswith(mcol.upper()):
                    if mcol not in df.columns:
                        df[mcol] = 0
                    df[mcol] += df[col]
                    to_drop.append(col)
        df = df.drop(columns=to_drop)
        if logger.isEnabledFor(logging.INFO):
            # why are 7 annotations empty?
            missing = ', '.join(map(str, df[df['text'].isna()].index.to_list()))
            logger.info(f"mimic missing annotations: {missing}")
        df = df[~df['text'].isna()]
        df = df.sort_values('note_id sentence_index'.split())
        df['id'] = df.apply(
            lambda r: f"{r['note_id']}.{r['sentence_index']}",
            axis=1).astype(str)
        assert len(df) == len(df['id'].drop_duplicates())
        df['labels'] = df.apply(map_row_labels, axis=1)
        return df

    def _get_synthetic_dataframe(self) -> pd.DataFrame:
        def format_labels(lb: str) -> str:
            return tuple(sorted(lb.lower().split(',')))

        par_dir: Path = self._get_parent_path()
        dfs: List[pd.DataFrame] = []
        fname: str
        for fname in self.synthetic_files:
            csv_file = par_dir / fname
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'loading: {csv_file}')
            dfs.append(pd.read_csv(csv_file))
        df = pd.concat(dfs, ignore_index=True)
        df['labels'] = df['label'].apply(format_labels)
        return df

    def _map_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        none: str = self.none_label
        df['labels'] = df['labels'].apply(
            lambda x: ('no',) if x == (none,) else ('yes',))
        return df

    def _get_mimic_binary_dataframe(self) -> pd.DataFrame:
        df: pd.DataFrame = self._get_mimic_dataframe()
        df = self._map_binary_labels(df)
        return df

    def _get_mimthetic_dataframe(self) -> pd.DataFrame:
        dfm: pd.DataFrame = self._get_mimic_dataframe()
        dfs: pd.DataFrame = self._get_synthetic_dataframe()
        dfm['src'] = 'mimic'
        dfm['src_id'] = dfm.index
        dfs['src'] = 'synthetic'
        dfs['src_id'] = dfs.index
        df: pd.DataFrame = pd.concat((dfm, dfs), ignore_index=True)
        df = df['src src_id text labels'.split()]
        return df

    def _get_mimthetic_binary_dataframe(self) -> pd.DataFrame:
        df: pd.DataFrame = self._get_mimthetic_dataframe()
        df = self._map_binary_labels(df)
        return df

    def _get_dataframe(self) -> pd.DataFrame:
        if self.split_col == 'mimic':
            return self._get_mimic_dataframe()
        elif self.split_col == 'synthetic':
            return self._get_synthetic_dataframe()
        elif self.split_col == 'mimic-binary':
            return self._get_mimic_binary_dataframe()
        if self.split_col == 'mimthetic':
            return self._get_mimthetic_dataframe()
        elif self.split_col == 'mimthetic-binary':
            return self._get_mimthetic_binary_dataframe()
        else:
            raise APIError(f'Unknown corpus split type: {self.split_col}')

    @persisted('_labels')
    def get_labels(self) -> Tuple[str, ...]:
        if self.split_col.endswith('-binary'):
            return ('no', 'yes')
        else:
            labels: List[str] = list(self.mimic_corpus_columns)
            labels.sort()
            labels.append(self.none_label)
            return tuple(labels)
