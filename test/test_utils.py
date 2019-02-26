import unittest
from collections import OrderedDict

from utils.utils import replace_morphemes
from utils.data_loader import DataLoader

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader()
        train_file = 'data/test.evn.train.ud'
        morphemes_path = 'data/morpheme/test.evn.train.morph'
        self.train = self.data_loader.load_conllu(train_file)
        self.morphemes = list(self.data_loader.load_morphemes(morphemes_path))

    def test_replace_morphemes(self):
        true_train = [
            OrderedDict([
                ('id', 1),
                ('form', 'tug'),
                ('lemma', 'tug'),
                ('upostag', 'ADV'),
                ('xpostag', None),
                ('feats', None),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)]),
            OrderedDict([
                ('id', 2),
                ('form', 'əɲinin'),
                ('lemma', 'əɲini'),
                ('upostag', 'NOUN'),
                ('xpostag', None),
                ('feats',
                 OrderedDict([
                    ('Case', 'Nom'),
                    ('Number', 'Sing'),
                    ('Poss', 'Yes'),
                    ('PossNumber', 'Sing'),
                    ('PossPerson', '3')])),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)]),
            OrderedDict([
                ('id', 3),
                ('form', 'nə'),
                ('lemma', 'nə'),
                ('upostag', 'PART'),
                ('xpostag', None),
                ('feats', None),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)]),
            OrderedDict([
                ('id', 4),
                ('form', 'əɲinin'),
                ('lemma', 'əɲini'),
                ('upostag', 'NOUN'),
                ('xpostag', None),
                ('feats',
                 OrderedDict([
                    ('Case', 'Nom'),
                    ('Number', 'Sing'),
                    ('Poss', 'Yes'),
                    ('PossNumber', 'Sing'),
                    ('PossPerson', '3')])),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)]),
            OrderedDict([
                ('id', 5),
                ('form', 'əː'),
                ('lemma', 'əː'),
                ('upostag', 'PRON'),
                ('xpostag', None),
                ('feats', OrderedDict([('Case', 'Nom'), ('Number', 'Sing')])),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)]),
            OrderedDict([
                ('id', 6),
                ('form', 'kəː'),
                ('lemma', 'kəː'),
                ('upostag', 'PART'),
                ('xpostag', None),
                ('feats', None),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)]),
            OrderedDict([
                ('id', 7),
                ('form', 'oːn'),
                ('lemma', 'oːn'),
                ('upostag', 'ADV'),
                ('xpostag', None),
                ('feats', None),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)]),
            OrderedDict([
                ('id', 8),
                ('form', 'ut͡ʃiťelɲitsa'),
                ('lemma', 'ut͡ʃiťelɲitsa'),
                ('upostag', 'X'),
                ('xpostag', None),
                ('feats', None),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)]),
            OrderedDict([
                ('id', 9),
                ('form', 'grit'),
                ('lemma', 'grit'),
                ('upostag', 'NOUN'),
                ('xpostag', None),
                ('feats', OrderedDict([('Case', 'Nom'), ('Number', 'Sing')])),
                ('head', None),
                ('deprel', '_'),
                ('deps', None),
                ('misc', None)])
        ]
        true_morphemes = [{'form': 'tug', 'morphemes': [{'label': 'ROOT', 'morpheme': 'tug'}]},
                           {'form': 'əɲinin',
                           'morphemes': [{'label': 'ROOT', 'morpheme': 'əɲini'},
                                         {'label': 'PS3SG', 'morpheme': 'n'}]},
                           {'form': 'nə',
                            'morphemes': [{'label': 'ROOT', 'morpheme': 'nə'}]},
                          {'form': 'əɲinin',
                           'morphemes': [{'label': 'ROOT', 'morpheme': 'əɲini'},
                                         {'label': 'PS3SG', 'morpheme': 'n'}]},
                          {'form': 'əː',
                           'morphemes': [{'label': 'ROOT', 'morpheme': 'əː'}]},
                           {'form': 'kəː',
                            'morphemes': [{'label': 'ROOT', 'morpheme': 'kəː'}]},
                          {'form': 'oːn', 'morphemes': [{'label': 'ROOT', 'morpheme': 'oːn'}]},
                          {'form': 'ut͡ʃiťelɲitsa',
                           'morphemes': [{'label': 'ROOT', 'morpheme': 'ut͡ʃiťelɲitsa'}]},
                          {'form': 'grit', 'morphemes': [{'label': 'ROOT', 'morpheme': 'grit'}]}]
        fact_train, fact_morphemes = replace_morphemes(self.train, self.morphemes)
        self.assertEqual(len(true_train), len(fact_train[0]))
        for i, w in enumerate(true_train):
            self.assertEqual(w, fact_train[0][i])
        self.assertEqual(true_morphemes, fact_morphemes[0])
