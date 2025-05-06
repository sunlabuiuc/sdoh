"""A class to reduce multilabel output.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import torch
from torch import Tensor
from zensols.deeplearn.domain import ModelSettings
from zensols.deeplearn.model import BatchIterator


class BinarySingleOutputBatchIterator(BatchIterator):
    def _encode_output(self, output: Tensor) -> Tensor:
        return super()._encode_output(output.squeeze(dim=1))

    def _encode_labels(self, labels: Tensor) -> Tensor:
        """Make labels 1-len vectors for the single neuron output."""
        labels = labels[:, 0]
        return super()._encode_labels(labels)


@dataclass
class BinaryHotCodeOutcomeReducer(object):
    """
    """
    model_settings: ModelSettings = field()
    """Configures the model."""

    def __call__(self, outcomes: Tensor) -> Tensor:
        """Fold back out to 2-len binary vectors."""
        outcomes.clamp_(min=0, max=1)
        outcomes.round_()
        outcomes = outcomes.squeeze(0).type(torch.LongTensor)
        outcomes = torch.nn.functional.one_hot(outcomes, num_classes=2)
        return outcomes
