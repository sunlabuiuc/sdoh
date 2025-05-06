"""Large language model inference and testing classes.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Dict, Set, Any, Type, ClassVar, Union, Iterable
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
import re
import os
from datetime import datetime
import textwrap as tw
from io import TextIOBase, StringIO
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.metrics as mt
import numpy as np
import pandas as pd
from datasets import Dataset
from zensols.config import Dictable
from zensols.util import Failure
from zensols.persist import (
    persisted, PersistedWork, Stash, UnionStash,
    PrimeableStash, ReadOnlyStash
)
from zensols.nlp import FeatureToken
from zensols.deepnlp.classify.multilabel import MultiLabelFeatureDocument
from zensols.datdesc import DataFrameDescriber
from zensols.dataset import DatasetSplitStash
from zensols.lmtask import TaskError, Task
from zensols.lmtask.instruct import InstructTaskRequest, InstructTask
from zensols.lmtask.dataset import LoadedTaskDatasetFactory

logger = logging.getLogger(__name__)


class SdohPredictionError(TaskError):
    def __init__(self, msg: str, key: str):
        super().__init__(f'{msg} for key {key}')


@dataclass
class DatasetFactory(LoadedTaskDatasetFactory):
    NONE_LABEL: ClassVar[str] = '-'
    feature_hint_type: str = field(default=None)
    temporary_dir: Path = field(default=None)
    swap_task_templates: bool = field(default=False)

    def __post_init__(self):
        self._dataframe = PersistedWork(
            path=self.temporary_dir / f'{self.model_identifier}-feature.dat',
            owner=self,
            mkdir=True)

    @property
    def model_identifier(self) -> str:
        return self.task.generator.resource.model_file_name

    @classmethod
    def map_labels(cls: Type, labels: Tuple[str, ...]) -> str:
        def map_label(lb: str) -> str:
            return none_lb if lb == FeatureToken.NONE else lb
        none_lb: str = cls.NONE_LABEL
        return tuple(map(map_label, labels))

    def _add_sdoh_cui(self, doc: MultiLabelFeatureDocument):
        text: str = doc.text
        ftnone: str = FeatureToken.NONE
        updated: bool = False
        for tok in doc.token_iter():
            sdoh: str = None if tok.sdoh_ == ftnone else tok.sdoh_
            concept: str = None if tok.cui_ == ftnone else tok.pref_name_
            if sdoh is not None or concept is not None:
                sio = StringIO()
                sio.write(tok.text)
                sio.write(' <')
                if sdoh is not None and sdoh:
                    sio.write("SDOH='")
                    sio.write(sdoh.replace('_', ' ').lower())
                    sio.write("'")
                if sdoh is not None and concept is not None:
                    sio.write(', ')
                if concept is not None:
                    sio.write("concept='")
                    sio.write(concept.lower())
                    sio.write("'")
                sio.write('>')
                tok.norm = sio.getvalue()
                updated = True
        if updated:
            text = doc.norm
        return text

    def _to_text(self, doc: MultiLabelFeatureDocument) -> str:
        ft: str = self.feature_hint_type
        text: str
        if ft == 'lm':
            text = doc.text
        elif ft == 'lmsdohcui':
            text = self._add_sdoh_cui(doc)
        else:
            raise TaskError(f'Unknown feature hint type: {ft}')
        return text

    @persisted('_dataframe')
    def _get_dataframe(self) -> pd.DataFrame:
        rows: List[Tuple[Any, ...]] = []
        stash: Stash = self.source
        key: str
        for key in stash.keys():
            doc: MultiLabelFeatureDocument = stash[key]
            labels: str = ','.join(self.map_labels(doc.labels))
            text: str = self._to_text(doc)
            rows.append((key, text, text, labels))
        return pd.DataFrame(rows, columns='id sent text labels'.split())

    def _prepare_dataset(self, ds: Dataset) -> Dataset:
        task: InstructTask = self.task
        train, inference = task.train_template, task.inference_template
        # hack to (over)reuse/abuse this class
        if self.swap_task_templates:
            task.train_template, task.inference_template = inference, train
        try:
            return super()._prepare_dataset(ds)
        finally:
            task.train_template, task.inference_template = train, inference

    def _create(self) -> Dataset:
        return self._from_source(self._get_dataframe())

    def clear(self):
        self._dataframe.clear()


@dataclass
class SplitDatasetFactory(DatasetFactory):
    split_stash: DatasetSplitStash = field(default=None, repr=False)

    @property
    def source(self) -> Stash:
        split_name: str = self.load_args['split']
        splits = self._create_train_test(self.split_stash)
        return splits[split_name]

    @source.setter
    def source(self, source: Any):
        if hasattr(self, '_dataframe'):
            raise ValueError('Read-only attribute: source')

    @persisted('_train_test_split')
    def _create_train_test(self, stash: DatasetSplitStash) -> Dict[str, Stash]:
        train = UnionStash((stash.splits['train'], stash.splits['validation']))
        test = stash.splits['test']
        if logger.isEnabledFor(logging.DEBUG):
            trainl = len(train)
            testl = len(test)
            tot: int = trainl + testl
            trainp = trainl / tot * 100
            testp = testl / tot * 100
            logger.debug(f'merged train / test: {trainl} ({trainp:.1f}%) / ' +
                         f'{testl} ({testp:.1f}%)')
        return {'train': train, 'test': test}

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_split_stash: bool = False):
        if isinstance(self.source, UnionStash):
            self._write_line('source:', depth, writer)
            stash: Stash
            for stash in self.source.stashes:
                self._write_object(stash, depth + 1, writer)
                #stash.split_container.write(depth, writer)
        else:
            self.source.write(depth, writer)
            #self.source.split_container.write(depth, writer)
        if include_split_stash:
            self._write_object(self.split_stash, depth, writer)
            self._write_object(self.split_stash.split_container, depth, writer)


@dataclass
class _ResponseParser(Dictable):
    _RESPONSE_REGEXS: ClassVar[Tuple[re.Pattern, ...]] = field(default=(
        re.compile(r'(?:.*?`([a-z,` ]{3,}`))', re.DOTALL),
        re.compile(r'.*?[`#-]([a-z, \t\n\r]{3,}?)[`-].*', re.DOTALL),
    ))
    labels: Tuple[str, ...] = field()
    none_label: str = field()

    def _match(self, text: str) -> str:
        for pat in self._RESPONSE_REGEXS:
            m: re.Match = pat.match(text)
            if m is not None:
                return m.group(1)

    def _reduce_labels(self, text: str) -> List[str]:
        parsed: Set[str] = set()
        lb: str
        for lb in self.labels:
            if text.find(lb) >= 0:
                parsed.add(lb)
        if len(parsed) > 0:
            return sorted(parsed)
        else:
            return [self.none_label]

    def __call__(self, text: str) -> str:
        matched: str = self._match(text)
        if matched is None:
            comma_labels = self.none_label
        else:
            labels: List[str] = self._reduce_labels(matched)
            comma_labels = ','.join(labels)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{text} -> {comma_labels}')
        return comma_labels


@dataclass
class _InferenceResource(Dictable):
    temporary_dir: Path = field()

    def get_temporary_file(self, name: str) -> Path:
        return self.temporary_dir / f'{self.factory.model_identifier}-{name}'


@dataclass
class ResponseStash(PrimeableStash, ReadOnlyStash, _InferenceResource):
    factory: DatasetFactory = field()

    def __post_init__(self):
        self._dataset = PersistedWork(
            path=self.get_temporary_file('ds.dat'),
            owner=self,
            mkdir=True)

    @property
    @persisted('_dataset')
    def dataset(self) -> Dict[str, Dict[str, Any]]:
        ds = {d['id']: d for d in self.factory.create()}
        # not true for mimic DS since some rows are skipped with empty sentences
        #assert set(ds.keys()) == set(map(str, range(len(ds))))
        return ds

    def load(self, name: str) -> Any:
        row: Dict[str, Any] = self.dataset.get(name)
        if row is not None:
            id: str = row['id']
            sent: str = row['sent']
            prompt: str = row['text']
            req = InstructTaskRequest(model_input=prompt)
            if logger.isEnabledFor(logging.INFO):
                msg: str = f'pid={os.getpid()}{id}:<<{tw.shorten(sent, 70)}>>'
                logger.info(f'processed: {msg}')
            response: str = self.factory.task.process(req).model_output
            now: datetime = datetime.now()
            return dict(zip('id time sent labels prompt response'.split(),
                            (id, now, sent, row['labels'], prompt, response)))

    def keys(self) -> Iterable[str]:
        return self.dataset.keys()

    def exists(self, name: str) -> bool:
        return name in self.dataset

    def clear(self):
        self._dataset.clear()


@dataclass
class ModelInferencer(_InferenceResource):
    factory: DatasetFactory = field()
    stash: Stash = field()
    labels: Tuple[str, ...] = field()
    data_name: str = field()
    data_desc: str = field()
    metrics_dir: Path = field()
    temporary_dir: Path = field()
    limit: int = field(default=sys.maxsize)

    def __post_init__(self):
        model_id: str = self.factory.model_identifier
        feattype: str = self.factory.feature_hint_type
        fname: str = f'{self.data_name}-{model_id}-{feattype}.dat'
        self._metrics = PersistedWork(
            path=self.metrics_dir / fname,
            owner=self,
            mkdir=True)

    @property
    def id(self) -> str:
        return f'{self.data_name}-{self.factory.feature_hint_type}'

    @property
    @persisted('_desc')
    def desc(self) -> str:
        task: Task = self.factory.task
        params = task.generator.resource.asdict()
        params['model_id'] = self.factory.model_identifier
        desc: str = self.data_desc.format(**params)
        return desc

    @property
    def prediction_file(self) -> Path:
        return self.get_temporary_file('preds.csv')

    @property
    def results_file(self) -> Path:
        return self._metrics.path

    def _create_dataframe(self) -> pd.DataFrame:
        fails: List[Failure] = []
        rows: List[Tuple[Any, ...]] = []
        v: Union[Failure, Any]
        for v in it.islice(self.stash.values(), self.limit):
            if isinstance(v, Failure):
                v.write_to_log(logger, logging.INFO)
                fails.append(v)
            else:
                rows.append(v)
        if len(fails) > 0:
            logger.warning(f'{len(rows)} failed to get a response')
        df = pd.DataFrame(rows)
        labels: Tuple[str, ...] = tuple(
            filter(lambda lb: lb != FeatureToken.NONE, self.labels))
        parser = _ResponseParser(labels, DatasetFactory.NONE_LABEL)
        logger.info(f'parsing: {self.prediction_file}...')
        df['preds'] = df['response'].apply(parser)
        df.index = df['id'].astype(int)
        df = df.drop(columns=['id'])
        df = df.sort_index()
        return df

    @property
    def dataframe(self) -> pd.DataFrame:
        pred_file: Path = self.prediction_file
        if pred_file.is_file():
            return pd.read_csv(pred_file, index_col=0)
        else:
            return self._create_dataframe()

    @property
    @persisted('_metrics')
    def metrics(self) -> pd.DataFrame:
        return self._get_metrics(self.dataframe)

    def _get_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        from zensols.deeplearn.result import PredictionsDataFrameFactory
        logger.info('calculating performance metrics')
        labels: Tuple[str, ...] = DatasetFactory.map_labels(self.labels)
        mlb = MultiLabelBinarizer()
        mlb.fit([sorted(labels)])
        golds: np.ndarray = mlb.transform(
            df['labels'].apply(lambda ls: ls.split(',')))
        preds: np.ndarray = mlb.transform(
            df['preds'].apply(lambda ls: ls.split(',')))
        mets: List[Union[str, float]] = []
        for pf, ave in (('w', 'weighted'), ('m', 'micro'), ('M', 'macro')):
            p, r, f, _ = mt.precision_recall_fscore_support(
                golds, preds, average=ave)
            mets.extend((f'{pf}F1', f, f'{pf}P', p, f'{pf}R', r))
        dfm = pd.DataFrame(
            data=[list(map(lambda i: mets[i * 2 + 1],
                           range(int(len(mets) / 2))))],
            columns=tuple(map(lambda i: mets[i * 2],
                              range(int(len(mets) / 2)))))
        setting, dataset = self.data_name.split('-')
        feattype: str = self.factory.feature_hint_type
        dfm['count'] = len(df)
        dfm['eval'] = [setting]
        dfm['dataset'] = [dataset]
        dfm['augmented'] = [feattype != 'lm']
        dfm['id'] = [self.id]
        dfm['desc'] = self.desc
        meta = list(PredictionsDataFrameFactory.METRIC_DESCRIPTIONS.items())
        meta.extend((
            ('eval', 'test setting'),
            ('augmented', 'whether the prompt was augmented with features'),
            ('id', 'unique identifier'),
            ('dataset', 'the dataset name'),
            ('desc', 'the dataset description')))
        return DataFrameDescriber(
            name=self.data_name,
            desc=self.desc,
            df=dfm,
            meta=tuple(meta))

    def save(self):
        self.dataframe.to_csv(self.prediction_file)
        logger.info(f'wrote: {self.prediction_file}')
        self.clear_metrics()
        self.metrics

    def clear_metrics(self):
        self._metrics.clear()

    def clear(self):
        self.factory.clear()
        if self.prediction_file.is_file():
            self.prediction_file.unlink()
        self.stash.clear()
