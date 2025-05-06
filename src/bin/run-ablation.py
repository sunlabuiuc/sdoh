#!/usr/bin/env python

"""Run the ablation tests.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Any
from dataclasses import dataclass, field
from pprint import pprint
import plac


@dataclass
class Runner(object):
    conf_file: str = field()
    dataset: str = field()
    gpu: int = field()

    def __call__(self, context: Dict[str, Any]):
        from zensols.cli import CliHarness
        from harness import create_harness

        harness: CliHarness = create_harness()
        args: List[str] = f'traintest -c {self.conf_file}'.split()
        context['sdoh_default'] = {'label': self.dataset}
        context['gpu_torch_config'] = {'cuda_device_index': self.gpu}
        harness.app_config_context = context
        print('running ablation...')
        pprint({'args': args, 'context': context})
        harness.execute(args)


@plac.annotations(
    model=('model type', 'positional', None, str, ['multilabel', 'binary']),
    dataset=('dataset', 'option', 'd', str, ['mimic', 'synthetic', 'mimthetic', 'mimthetic']),
    gpu=('GPU index', 'option', 'g', int))
def main(model: str, dataset: str = 'mimic', gpu: int = 0):
    """Train and test the model for ablation experiments."""
    if model == 'binary':
        dataset = f'{dataset}-binary'
    runner = Runner(
        conf_file=f'etc/{model}.conf',
        dataset=dataset,
        gpu=gpu)
    batch_no_cui = {
        'decoded_attributes':
        'set: labels, transformer_enum_expander, transformer_trainable_embedding'
    }
    batch_with_cui = {
        'decoded_attributes':
        'set: labels, cuidescs_expander, transformer_enum_expander, transformer_trainable_embedding'
    }
    runner({'batch_stash': {
        'decoded_attributes':
        'set: labels, transformer_trainable_embedding'}})
    batch_config: Dict[str, Any]
    for batch_config in (batch_no_cui, batch_with_cui):
        runner({
            'batch_stash': batch_config,
            'enum_feature_vectorizer': {
                'decoded_feature_ids': 'set: tag, dep'}})
        runner({
            'batch_stash': batch_config,
            'enum_feature_vectorizer': {
                'decoded_feature_ids': 'set: tag, dep, ent'}})
        runner({
            'batch_stash': batch_config,
            'enum_feature_vectorizer': {
                'decoded_feature_ids': 'set: tag, dep, ent, medent'}})
        runner({
            'batch_stash': batch_config,
            'enum_feature_vectorizer': {
                'decoded_feature_ids': 'set: tag, dep, ent, medent, sdoh_'}})


if (__name__ == '__main__'):
    plac.call(main)
