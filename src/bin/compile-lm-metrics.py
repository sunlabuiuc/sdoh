#!/usr/bin/env python

from typing import Tuple, List, Iterable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import itertools as it
import pickle
import warnings
import pandas as pd
from zensols.config import ConfigFactory
from zensols.cli import CliHarness; CliHarness.add_sys_path('src/python')
from zensols import deepnlp; deepnlp.init()
from zensols.datdesc import DataFrameDescriber, DataDescriber
from zensols.sdoh import ApplicationFactory
from zensols.sdoh.lm import ModelInferencer

logger = logging.getLogger(__name__)


@dataclass
class MetricsCompiler(object):
    data_dir: Path = field(default=Path('data'))

    def __post_init__(self):
        from sklearn.exceptions import UndefinedMetricWarning
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        self.founds: List[Tuple[Path, ModelInferencer]] = []
        self.missing: List[Path] = []

    def _clean_stale(self):
        stale_conf: Path = self.data_dir / 'app-config.dat'
        if stale_conf.is_file():
            logger.info(f'removing {stale_conf}...')
            stale_conf.unlink()

    def _create_args(self, dataset: str, large_model: bool,
                     feattype: str) -> List[str]:
        args: List[str] = ['-c', 'etc/model.conf']
        override: str = (f'sdoh_default.label={dataset},' +
                         f'sdoh_lm_default.feature_hint_type={feattype}')
        if large_model:
            model_id = 'meta-llama/Llama-3.3-70B-Instruct'
            override += (f',lmtask_trainer_default.source_model={model_id},' +
                         f'lmtask_llama_instruct_resource.model_id={model_id}')
        args.append(f'--override={override}')
        return args

    def _create_factory(self, dataset: str, large_model: bool,
                        feattype: str) -> ConfigFactory:
        args: List[str] = self._create_args(dataset, large_model, feattype)
        harn: CliHarness = ApplicationFactory.create_harness()
        return harn.get_config_factory(args)

    def _get_configs(self) -> Iterable[ConfigFactory]:
        datasets: List[str] = 'mimic synthetic mimthetic'.split()
        tests: List[str] = 'fewshot test'.split()
        model_size: List[bool] = (False, True)
        feattypes: List[str] = 'lm lmsdohcui'.split()
        for dataset, test_name, size, feattype in it.product(
                datasets, tests, model_size, feattypes):
            logger.info(f'parsing={dataset}, test={test_name}, 70B={size}, feattype={feattype}')
            yield {'config_factory': self._create_factory(dataset, size, feattype),
                   'test_name': test_name,
                   'meta': {'dataset': dataset,
                            'test': test_name,
                            'large_model': size,
                            'feattype': feattype}}

    def _reparse_results(self) -> Iterable[Path]:
        for config in self._get_configs():
            fac: ConfigFactory = config['config_factory']
            test_name: str = config['test_name']
            sec: str = f'sdoh_lm_{test_name}_inferencer'
            mi: ModelInferencer = fac(sec)
            pred_file: Path = mi.prediction_file
            if pred_file.is_file():
                self.founds.append((pred_file, mi, config['meta']))
                mi.save()
                yield mi.results_file
            else:
                self.missing.append(pred_file)

    def _reparse_into_data_describer(self) -> DataDescriber:
        result_files: Tuple[Path, ...] = tuple(self._reparse_results())
        dfds: List[DataFrameDescriber] = []
        path: Path
        for path in result_files:
            with open(path, 'rb') as f:
                dfd: DataFrameDescriber = pickle.load(f)
                assert isinstance(dfd, DataFrameDescriber)
                dfds.append(dfd)
        return DataDescriber(tuple(dfds), name='language model predictions')

    def _get_analyzer_file(self) -> Path:
        fac: ConfigFactory = next(iter(self._get_configs()))['config_factory']
        analyzer_sec = fac.config['sdoh_analyzer']
        met_file: Path = analyzer_sec.lm_metrics_file
        met_file.parent.mkdir(parents=True, exist_ok=True)
        return met_file

    def _compile_predictions(self) -> pd.DataFrame:
        dfs: List[pd.DataFrame] = []
        path: Path
        mi: ModelInferencer
        for path, mi, meta in self.founds:
            df: pd.DataFrame = pd.read_csv(path)
            if 'time' not in df.columns:
                df['time'] = None
            df = df['id time labels preds'.split()]
            df['desc'] = mi.desc
            df.insert(0, 'ds_id', mi.id)
            for k, v in meta.items():
                df[k] = v
            dfs.append(df)
        return pd.concat(dfs)

    def compile(self):
        self._clean_stale()
        self.missing.clear()
        dd: DataDescriber = self._reparse_into_data_describer()
        results = {'summary': dd,
                   'predictions': self._compile_predictions()}
        analyzer_met_file = self._get_analyzer_file()
        with open(analyzer_met_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f'wrote: {analyzer_met_file}')
        for path in self.missing:
            print(f'missing results: {path}')


def main():
    logging.basicConfig(
        format='%(asctime)s[%(levelname)s]: %(message)s',
        level=logging.WARNING)
    logger.setLevel(logging.DEBUG)
    comp = MetricsCompiler()
    comp.compile()


if (__name__ == '__main__'):
    main()
