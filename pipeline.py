import time
import uuid
import pickle
import os.path
import logging

import numpy as np
from sklearn.model_selection import KFold
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from data_loader import DataLoader
from feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)

class Pipeline:
    def __init__(self, lang_prefix):
        self.lang_prefix = lang_prefix
        data_path = 'data'
        self.train_file = data_path + '/{}.train.ud'.format(self.lang_prefix)

        assert os.path.exists(data_path), 'There is no {} directory'.format(data_path)
        assert os.path.exists(self.train_file), 'There is no {} directory'.format(self.train_file)

        self.data_loader = DataLoader()
        self.feature_extractor = FeatureExtractor()
        self.clfr_pos = sklearn_crfsuite.CRF()

    def pickle_model(self):
        with open('models/{}_pos.pkl'.format(self.lang_prefix), 'wb') as f:
            pickle.dump(self.clfr_pos, f)

    def pipeline(self):
        train = self.data_loader.load_conllu(self.train_file)
        train = self.feature_extractor.del_hyphen_parts(train)
        y_true = [self.feature_extractor.sent2labels(sent, category=None, pos=True) for sent in train]  # классы - части речи
        logging.info('Loaded train and test')

        kf = KFold(n_splits=5)
        y_pred = np.empty_like(train)
        fold_count = 0
        for train_index, test_index in kf.split(train):
            logging.info('Fold {}'.format(fold_count))
            fold_train, fold_test = [train[i] for i in train_index], [train[i] for i in test_index]

            X_train = [self.feature_extractor.sent2features(sent, sent_id) for sent_id, sent in enumerate(fold_train)]
            y_train = [self.feature_extractor.sent2labels(sent, category=None, pos=True) for sent in fold_train]  # классы - части речи
            logging.info('Train features are extracted')

            X_test = [self.feature_extractor.sent2features(sent, sent_id) for sent_id, sent in enumerate(fold_test)]
            logging.info('Test features are extracted')

            logging.info('Training classifier...')
            self.clfr_pos.fit(X_train, y_train)

            logging.info('Saving classifier...')
            self.pickle_model()

            logging.info('Prediction...')
            y_pred_fold = self.clfr_pos.predict(X_test)
            y_pred[test_index] = y_pred_fold
            fold_count += 1

        # print('Metrics for all labels:')
        # print(metrics.flat_classification_report(
        #     y_true, y_pred
        # ))
        #
        labels = self.clfr_pos.classes_.copy()
        labels.remove('X')

        print('Lang {}. Metrics for target labels (without "X" label):'.format(self.lang_prefix))
        print(metrics.flat_classification_report(
            y_true, y_pred, labels=labels
        ))


if __name__ == '__main__':
    pipeline = Pipeline(lang_prefix='evn')
    pipeline.pipeline()
    pipeline = Pipeline(lang_prefix='vep')
    pipeline.pipeline()
    pipeline = Pipeline(lang_prefix='sel')
    pipeline.pipeline()
    pipeline = Pipeline(lang_prefix='krl')
    pipeline.pipeline()
    pipeline = Pipeline(lang_prefix='lud')
    pipeline.pipeline()
    pipeline = Pipeline(lang_prefix='olo')
    pipeline.pipeline()
