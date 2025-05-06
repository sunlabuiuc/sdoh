#!/usr/bin/env python

from typing import List
from zensols.cli import ConfigurationImporterCliHarness
from zensols import deepnlp

# initialize the NLP system
deepnlp.init()


def clear_config_cache():
    from pathlib import Path
    config_cache = Path('data/app-config.dat')
    if config_cache.is_file():
        config_cache.unlink()


def create_harness(proto_args: List[str] = []):
    return ConfigurationImporterCliHarness(
        src_dir_name='src/python',
        app_factory_class='zensols.sdoh.ApplicationFactory',
        config_path='etc/model.conf',
        # remember to clear config cache if label changes
        proto_args=proto_args,
        proto_factory_kwargs={'reload_pattern': r'^(?:zensols.sdoh)'}
    )


if (__name__ == '__main__'):
    if 0:
        clear_config_cache()
    action: List[str] = ['proto']
    args: List[str] = action + ['-c', 'etc/model.conf']
    model_id: str = None
    dataset: str = 'mimic' if 1 else 'synthetic'
    feattype: str = 'lm' if 1 else 'lmsdohcui'
    override: str = (f'sdoh_default.label={dataset},' +
                     f'sdoh_lm_default.feature_hint_type={feattype}')
    if model_id is not None:
        override += f',lmtask_trainer_default.source_model={model_id}'
        override += f',lmtask_llama_instruct_resource.model_id={model_id}'
    args.append(f'--override={override}')
    harness: ConfigurationImporterCliHarness = create_harness(args)
    harness.run()
