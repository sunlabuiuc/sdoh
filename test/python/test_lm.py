import logging
import unittest
import pandas as pd
from zensols.sdoh.lm import _ResponseParser as ResponseParser

if 0:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)


class Test(unittest.TestCase):
    def setUp(self):
        lb = 'transportation housing relationship employment support parent'
        self.parser = ResponseParser(
            labels=tuple(sorted(lb.split())),
            none_label='-')
        self.df: pd.DataFrame = pd.read_csv('test-resources/response-parse.csv')

    def test_respons_parse(self):
        par: ResponseParser = self.parser
        df: pd.DataFrame = self.df
        for i, should, text in df.itertuples(index=True, name=None):
            lbs: str = par(text)
            self.assertEqual(should, lbs,
                             f'{i}: <<{should}>>==<<{lbs}>> <~{text}~>')
