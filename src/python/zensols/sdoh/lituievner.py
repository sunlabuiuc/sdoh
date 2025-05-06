"""A vectorizer for the SDoH NER.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict, Sequence, Union
from dataclasses import dataclass, field
from frozendict import frozendict
from spacy.language import Language
from spacy.tokens import Span
from zensols.persist import persisted
from zensols.config import ConfigFactory
from zensols.nlp.sparser import SpacyFeatureDocumentParser
from zensols.deepnlp.vectorize import SpacyFeatureVectorizer


@dataclass
class SdohFeatureDocumentParser(SpacyFeatureDocumentParser):
    """This fixes the issue of two (MedCAT and the Lituievner el al.) spaCy span
    extensions stepping on eachothers' feet.

    This fixes the issue by removing the ``cui`` extension added by MedCAT since
    it will be added again in the sdoh NER component.  This is a class space
    attribute that *should* be sharble by both.

    """
    def _create_model(self) -> Language:
        if Span.get_extension('cui') is not None:
            Span.remove_extension('cui')
        return super()._create_model()


@dataclass(repr=False, init=False)
class SdohSpacyFeatureVectorizer(SpacyFeatureVectorizer):
    """Vectorizes SDoH features.  Note that this vectorizer needs to use the
    ``sdoh_`` (note the underscore) as the feature name.  This is because the
    reverse lookup on the :obj:`spacy.vocab.Vocab.strings` won't work as they
    appear to be added when the parser sees classes it hasn't yet predicted for
    the lifecycle of the Python interpreter.

    Citation:

      `Lituiev et al. (2023)`_ Automatic extraction of social determinants of
      health from medical notes of chronic lower back pain patients

    .. _Lituiev et al. (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10354762

    """
    second_level: bool = field()
    """Whether to use the second level models' labels (see the paper)."""

    def __init__(self, name: str, config_factory: ConfigFactory,
                 second_level: bool, *args, **kwargs):
        self.model = kwargs['model']
        self.second_level = second_level
        symbols = self.second_level_labels \
            if self.second_level \
            else self.first_level_labels
        super().__init__(
            *args,
            name=name,
            config_factory=config_factory,
            symbols=symbols,
            **kwargs)

    def _initialize(self, symbols: Union[str, Sequence[str]]):
        super()._initialize(symbols)
        self.symbol_to_vector: Dict[str, int] = dict(self.symbol_to_vector)
        self.symbol_to_norm: Dict[str, float] = dict(self.symbol_to_norm)
        if not self.second_level:
            l1_key: str
            l2_labels: Dict[str, Tuple[str, ...]]
            for l1_key, l2_labels in self.labels.items():
                for l2_key in l2_labels:
                    long_lb: str = f'{l1_key}: {l2_key}'
                    self.symbol_to_vector[long_lb] = \
                        self.symbol_to_vector[l1_key]
                    self.symbol_to_norm[long_lb] = \
                        self.symbol_to_norm[l1_key]
        self.symbol_to_vector = frozendict(self.symbol_to_vector)
        self.symbol_to_norm = frozendict(self.symbol_to_norm)

    @property
    @persisted('_first_level_labels')
    def first_level_labels(self) -> Tuple[str, ...]:
        """The labels used by the SDoH NER prediction model for just the first
        level models (see the paper).

        """
        return tuple(self.labels.keys())

    @property
    @persisted('_second_level_labels')
    def second_level_labels(self) -> Tuple[str, ...]:
        """The labels used by the SDoH NER prediction model for models across
        both levels.

        """
        labels: List[str] = []
        l1_key: str
        l2_labels: Dict[str, Tuple[str, ...]]
        for l1_key, l2_labels in self.labels.items():
            for l2_key in l2_labels:
                labels.append(f'{l1_key}: {l2_key}')
        return tuple(labels)

    @property
    @persisted('_labels')
    def labels(self) -> Dict[str, Tuple[str, ...]]:
        """The labels used by the SDoH NER prediction model."""
        from en_sdoh_bow.sdoh_bow import Level2Predictor
        labels: Dict[str, Tuple[str, ...]] = {}
        pipe: Level2Predictor = self.model.get_pipe('sdoh_bow')
        for l1_key, l1_model in pipe.classification_models.items():
            labels[l1_key] = tuple(l1_model.classes_)
        return frozendict(labels)
