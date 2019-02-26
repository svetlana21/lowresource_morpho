import os
import logging
from collections import OrderedDict
from argparse import ArgumentParser

from utils.utils import check_form_to_morpheme, init_logging
from utils.data_loader import DataLoader
from pipeline.feature_extractor import FeatureExtractor
from utils.morpheme_preprocessor import MorphemePreprocessor


class Inference:
    def __init__(self, lang_prefix, path_to_save, add_morpheme_features=False):
        self.lang_prefix = lang_prefix
        self.path_to_save = path_to_save
        self.add_morpheme_features = add_morpheme_features
        self.data_loader = DataLoader()
        self.feature_extractor = FeatureExtractor()

        self.test_file = 'test_data/{}.test.ud'.format(self.lang_prefix)
        self.morphemes_path = 'test_data/morpheme/{}.test.morph'.format(self.lang_prefix)
        self.models_path = 'models/{}'.format(self.lang_prefix)
        self.morphemes_index_path = 'models/{}/{}_morphemes.pkl'.format(self.lang_prefix, self.lang_prefix)

        self.categories = self.data_loader.load_json('data/grammar_data/{}_categories.json'.format(self.lang_prefix))
        self.pos2categories = self.data_loader.load_json(
            'data/grammar_data/{}_pos2categories.json'.format(self.lang_prefix))

        init_logging('inference')

    def inference(self):
        test_data = self.data_loader.load_non_labeled(self.test_file)
        logging.info('Test file {} is loaded'.format(self.test_file))

        if self.add_morpheme_features:
            logging.info('Morphemes {} preprocessing...'.format(self.morphemes_path))
            morphemes = list(self.data_loader.load_morphemes(self.morphemes_path))
            labels2ind = self.data_loader.load_model(self.morphemes_index_path)
            check_form_to_morpheme(test_data, morphemes)
            morpheme_preprocessor = MorphemePreprocessor(lang_prefix=self.lang_prefix, morphemes=morphemes,
                                                         labels2ind=labels2ind)
            self.feature_extractor.set_morpheme_preproc(morpheme_preproc=morpheme_preprocessor)

        logging.info('Feature extraction...')
        X_test = [self.feature_extractor.sent2features(sent, sent_id) for sent_id, sent in enumerate(test_data)]

        pos_model = self.data_loader.load_model(os.path.join(self.models_path, '{}_pos.pkl'.format(self.lang_prefix)))

        logging.info('POS prediction...')
        pos_pred = pos_model.predict(X_test)  # определение постэгов слов в тестовой выборке

        # добавление полученных постэгов в качестве признаков для моделей, распознающих грам. категории
        X_test_new = self.feature_extractor.add_pos_features(X_test, pos_pred)

        pred_categories = self.pred_categories_dict_initializer(X_test_new)

        for category in self.categories:
            logging.info('{} grammar category prediction...'.format(category))
            gc_model = self.data_loader.load_model(os.path.join(
                self.models_path, '{}_{}.pkl'.format(self.lang_prefix, category)))

            for i, sent in enumerate(X_test_new):
                for j, sample in enumerate(sent):
                    if sample['postag'] in self.pos2categories:
                        if category in self.pos2categories[sample['postag']]:
                            prediction = gc_model.predict([[sample]])[0][0]
                            if prediction != 'O':
                                pred_categories[i][j][category] = prediction

        result_test = self.add_tags(test_data, pos_pred, pred_categories)
        logging.info('Writing to {}...'.format(self.path_to_save))
        self.writing(result_test, self.path_to_save)

    def pred_categories_dict_initializer(self, test):
        """
        Инициализация пустого словаря, который будет заполнен предсказанными значениями грамматических категорий
        :param test:
        :return:
        """
        pred_categories = OrderedDict()
        for i, sent in enumerate(test):
            pred_categories[i] = OrderedDict()
            for j, sample in enumerate(sent):
                pred_categories[i][j] = OrderedDict()
        return pred_categories

    def add_tags(self, result_test, pos_pred, pred_categories):
        """
        Теггер тестовой выборки:
        вставка предсказанных тегов на их место в словарь, полученный после парсинга тестовой выборки.
        """

        for sent_i, sent in enumerate(result_test):
            sent_pos_labels = pos_pred.pop(0)  # частеречные теги для одного предложения
            sent_gc_labels = pred_categories[sent_i]     # теги грам. категорий для одного предложения
            for word_i, word in enumerate(sent):
                word['id'] = word_i + 1
                word['upostag'] = sent_pos_labels[word_i]  # добавить ключ 'upostag'
                word['feats'] = sent_gc_labels[word_i]  # добавление ключа 'feats'
        return result_test

    def writing(self, results, filename):
        """
        Запись в файл полученных результатов.
        """
        with open(filename, 'w', encoding='utf-8') as result:
            for sent in results:
                for word in sent:
                    result.write('{}\t{}\t_\t{}\t_\t'.format(word['id'], word['form'], word['upostag']))
                    if word['feats']:
                        keys_list = word['feats'].keys()
                        for i, key in enumerate(keys_list):
                            if i < len(keys_list) - 1:
                                result.write('{}={}|'.format(key, word['feats'][key]))
                            else:
                                result.write('{}={}\t_\t_\t_\t_\n'.format(key, word['feats'][key]))
                    else:
                        result.write('_\n')
                result.write('\n')
        logging.info('Finished')

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--lang', type=str, required=True)
    arg_parser.add_argument('--save-to', dest='path_to_save', type=str, required=True)
    arg_parser.add_argument('--morphemes', dest='add_morpheme_features', default=False, type=bool, required=False,
                            help='Add morpheme features')
    args = arg_parser.parse_args()

    inference_object = Inference(args.lang, args.path_to_save, args.add_morpheme_features)
    inference_object.inference()
