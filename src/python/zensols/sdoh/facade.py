"""Application facade overrides default behavior.

"""
__author__ = 'Paul Landes'

from typing import List, Union, Type
from dataclasses import dataclass, field
import logging
import pandas as pd
from zensols.datdesc import DataFrameDescriber
from zensols.deeplearn.model import ModelResult, ModelResultReporter
from zensols.deepnlp.classify import MultilabelClassifyModelFacade


@dataclass
class SdohModelResultReporter(ModelResultReporter):
    def _create_data_frame_describer(self, *args, **kwargs) \
            -> DataFrameDescriber:
        col_name: str = 'enum_features'
        dfd = super()._create_data_frame_describer(*args, **kwargs)
        df: pd.DataFrame = dfd.df
        cpos: int = df.columns.to_list().index('features')
        arch_fids: List[str] = []
        for fname, ar in self._get_archive_results():
            mr: ModelResult = ar.model_result
            dec_attribs: List[str] = \
                mr.config['batch_stash']['decoded_attributes']
            has_cui_emb: bool = 'cuidescs_expander' in dec_attribs
            fids: List[str] = list(
                mr.config['enum_feature_vectorizer'].get('decoded_feature_ids'))
            if has_cui_emb:
                fids.append('cui_embedding')
            fids.sort()
            arch_fids.append(None if fids is None else ', '.join(fids))
        assert len(arch_fids) == len(df)
        df.insert(cpos + 1, col_name, arch_fids)
        row = pd.Series(['linguistic token feature'],
                        index=[col_name],
                        name='description')
        dfd.meta = pd.concat((dfd.meta, row.to_frame()))
        return dfd


@dataclass
class SdohModelFacade(MultilabelClassifyModelFacade):
    model_result_reporter_class: Union[str, Type[ModelResultReporter]] = \
        field(default=SdohModelResultReporter)

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for i in ['zensols.sdoh.model']:
            logging.getLogger(i).setLevel(logging.DEBUG)
