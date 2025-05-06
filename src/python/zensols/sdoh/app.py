"""A model that predicts Social Determinants of Health.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
from zensols.config import ConfigFactory

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """A model that predicts social determinates of health.

    """
    config_factory: ConfigFactory = field()

    def _inference(self, name: str, clear: bool, max_sents: int):
        from .lm import ModelInferencer
        mi: ModelInferencer = self.config_factory(name)
        if clear:
            mi.clear()
        if max_sents is not None:
            mi.limit = max_sents
        mi.factory.task.write()
        mi.save()
        mi.metrics.write()

    def few_shot_process(self, clear: bool = False, max_sents: int = None):
        """Predict SDOHs on the configured corpus as few-shot examples.

        :param clear: whether to first clear previous results

        :param max_sents: the max number of sentences to process

        """
        self._inference('sdoh_lm_fewshot_inferencer', clear, max_sents)

    def test_process(self, clear: bool = False, max_sents: int = None):
        """Predict SDOHs on the configured test corpus.

        :param clear: whether to first clear previous results

        :param max_sents: the max number of sentences to process

        """
        self._inference('sdoh_lm_test_inferencer', clear, max_sents)
