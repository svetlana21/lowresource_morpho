import os.path
import logging
from pathlib import Path
from argparse import ArgumentParser

from sklearn.model_selection import KFold
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from utils.utils import replace_morphemes, invert_dict, init_logging, get_categories
from utils.data_loader import DataLoader
from utils.morpheme_preprocessor import MorphemePreprocessor
from pipeline.feature_extractor import FeatureExtractor


class Pipeline:
    def __init__(self, lang_prefix, add_morpheme_features=False):
        self.lang_prefix = lang_prefix
        self.add_morpheme_features = add_morpheme_features

        data_path = str(Path(__file__).parents[1]) + '/data'
        self.train_file = data_path + '/{}.train.ud'.format(self.lang_prefix)
        if self.add_morpheme_features:
            self.morphemes_path = data_path + '/morpheme/{}.train.morph'.format(self.lang_prefix)
            assert os.path.exists(self.morphemes_path), 'There is no {} directory'.format(self.morphemes_path)

        assert os.path.exists(data_path), 'There is no {} directory'.format(data_path)
        assert os.path.exists(self.train_file), 'There is no {} directory'.format(self.train_file)

        self.data_loader = DataLoader()
        self.feature_extractor = FeatureExtractor()
        self.clfr_pos = sklearn_crfsuite.CRF(all_possible_transitions=True)

        categories_path = data_path + '/grammar_data/{}_categories.json'.format(self.lang_prefix)
        pos2categories = data_path + '/grammar_data/{}_pos2categories.json'.format(self.lang_prefix)

        assert os.path.exists(categories_path), \
            'There is no {} file (create it with utils.utils.get_categories)'.format(categories_path)
        assert os.path.exists(pos2categories), \
            'There is no {} file (create it with utils.utils.get_categories)'.format(pos2categories)

        self.categories = self.data_loader.load_json(categories_path)
        self.pos2categories = self.data_loader.load_json(pos2categories)
        self.categories2pos = invert_dict(self.pos2categories)

        init_logging('pipeline')

        train = self.data_loader.load_conllu(self.train_file)

        self.morphemes = None
        self.morpheme_preproc = None

        if self.add_morpheme_features and self.lang_prefix in {'evn', 'sel'}:
            morphemes = list(self.data_loader.load_morphemes(self.morphemes_path))
            self.morpheme_preproc = MorphemePreprocessor(lang_prefix=self.lang_prefix, morphemes=morphemes)
            # приведение к соответствию (или проверка на соответствие) токенов train'а и морфемной сегментации
            self.train, self.morphemes = replace_morphemes(train, morphemes)
        else:
            # удаление из train'а токенов дефисных написаний (с id-шниками типа 1-2 и без тегов)
            self.train = self.feature_extractor.del_hyphen_parts(train)

    def get_features_for_gc_classfier(self, fold, category, morphemes_fold=None):
        X, y = [], []
        if morphemes_fold:
            self.feature_extractor.set_morphemes_fold(morphemes_fold)
        for sent_id, sent in enumerate(fold):
            sent_feats, sent_labels = self.feature_extractor.sent2features_gc(
                sent, sent_id, category, self.categories2pos[category])
            if sent_feats:
                X.append(sent_feats)
                y.append(sent_labels)
        return X, y

    def pipeline_cv(self, categories=True):
        kf = KFold(n_splits=5)
        fold_count = 0
        logging.info('Lang: {}'.format(self.lang_prefix))
        for train_index, test_index in kf.split(self.train):
            logging.info('Fold {}'.format(fold_count))
            fold_train, fold_test = [self.train[i] for i in train_index], [self.train[i] for i in test_index]
            morphemes_train, morphemes_test = None, None

            if self.add_morpheme_features:
                # морфемы тоже делим на фолды
                morphemes_train, morphemes_test = [self.morphemes[i] for i in train_index], \
                                                  [self.morphemes[i] for i in test_index]
                self.feature_extractor.set_morpheme_preproc(self.morpheme_preproc)
                self.feature_extractor.set_morphemes_fold(morphemes_train)

            X_train = [self.feature_extractor.sent2features(sent, sent_id) for sent_id, sent in enumerate(fold_train)]
            y_train = [self.feature_extractor.sent2labels(sent, category=None, pos=True) for sent in fold_train]

            if self.add_morpheme_features:
                self.feature_extractor.set_morphemes_fold(morphemes_test)
            X_test = [self.feature_extractor.sent2features(sent, sent_id) for sent_id, sent in enumerate(fold_test)]
            y_test = [self.feature_extractor.sent2labels(sent, category=None, pos=True) for sent in fold_test]

            # logging.info('Training POS classifier...')
            # self.clfr_pos.fit(X_train, y_train)
            # y_pred = self.clfr_pos.predict(X_test)
            #
            # # Из оценки исключаем UNKN-классы
            # labels = self.clfr_pos.classes_.copy()
            # if 'X' in labels:
            #     labels.remove('X')
            #
            # logging.info('Metrics for target labels (without "X" label):')
            # logging.info(metrics.flat_classification_report(
            #     y_test, y_pred, labels=labels
            # ))

            if categories:
                # цикл, создающий модели для грам. категорий
                for category in self.categories:
                    X_train, y_train = self.get_features_for_gc_classfier(fold=fold_train, category=category,
                                                                          morphemes_fold=morphemes_train)

                    X_test, y_test = self.get_features_for_gc_classfier(fold=fold_test, category=category,
                                                                        morphemes_fold=morphemes_test)

                    logging.info('Training classifier for {} category...'.format(category))
                    clfr = sklearn_crfsuite.CRF(all_possible_transitions=True)
                    clfr.fit(X_train, y_train)
                    y_pred = clfr.predict(X_test)

                    logging.info('Metrics for {} grammar category:'.format(category))
                    logging.info(metrics.flat_classification_report(
                        y_test, y_pred
                    ))

            fold_count += 1

    def pipeline_train(self, categories=True):
        logging.info('Lang: {}'.format(self.lang_prefix))

        if self.add_morpheme_features:
            self.feature_extractor.set_morpheme_preproc(self.morpheme_preproc)
            self.feature_extractor.set_morphemes_fold(self.morphemes)

        X_train = [self.feature_extractor.sent2features(sent, sent_id) for sent_id, sent in enumerate(self.train)]
        y_train = [self.feature_extractor.sent2labels(sent, category=None, pos=True) for sent in self.train]

        # logging.info('Training POS classifier...')
        # self.clfr_pos.fit(X_train, y_train)
        # self.data_loader.pickle_model(lang=self.lang_prefix, task='pos', model=self.clfr_pos)

        if categories:
            # цикл, создающий модели для грам. категорий
            for category in self.categories:
                X_train, y_train = self.get_features_for_gc_classfier(fold=self.train, category=category,
                                                                      morphemes_fold=self.morphemes)

                logging.info('Training classifier for {} category...'.format(category))
                clfr = sklearn_crfsuite.CRF(all_possible_transitions=True)
                clfr.fit(X_train, y_train)
                self.data_loader.pickle_model(lang=self.lang_prefix, task=category, model=clfr)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--lang', type=str, required=True)
    arg_parser.add_argument('--morphemes', dest='add_morpheme_features', default=False, type=bool, required=False,
                            help='Add morpheme features')
    arg_parser.add_argument('--option', type=str, required=True, choices=['cv', 'train'])
    arg_parser.add_argument('--categories', default=True, type=bool, required=False)
    args = arg_parser.parse_args()

    pipeline = Pipeline(lang_prefix=args.lang, add_morpheme_features=args.add_morpheme_features)

    if args.option == 'train':
        pipeline.pipeline_train(categories=args.categories)
    elif args.option == 'cv':
        pipeline.pipeline_cv(categories=args.categories)
    else:
        logging.error('Unknown option {}'.format(args.option))
