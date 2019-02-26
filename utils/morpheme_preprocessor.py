import os
import pickle
import numpy as np


class MorphemePreprocessor:
    def __init__(self, lang_prefix, morphemes: list, labels2ind=None):
        self.lang_prefix = lang_prefix
        self.morphemes = morphemes
        if labels2ind:
            self.labels = sorted(labels2ind.keys())
            self.label2ind = labels2ind
        else:
            self.labels = self.morpheme_classes()
            self.label2ind = self.labels_index(self.labels)
            self.pickle_model()

    def morpheme_classes(self):
        labels = set()
        for sent in self.morphemes:
            for word in sent:
                for morpheme in word['morphemes']:
                    if morpheme['label'] != 'ROOT':
                        labels.add(morpheme['label'])
        return labels

    def labels_index(self, labels):
        self.labels = sorted(labels)
        return {label: i for i, label in enumerate(labels)}

    def one_hot(self, word_form):
        empty_vector = np.zeros_like(self.labels, dtype=int)
        for m in word_form['morphemes']:
            if m['label'] != 'ROOT':
                empty_vector[self.label2ind[m['label']]] = 1
        return empty_vector

    def pickle_model(self):
        if not os.path.exists('models/{}'.format(self.lang_prefix)):
            os.mkdir('models/{}'.format(self.lang_prefix))
        with open('models/{}/{}_morphemes.pkl'.format(self.lang_prefix, self.lang_prefix), 'wb') as f:
            pickle.dump(self.label2ind, f)
