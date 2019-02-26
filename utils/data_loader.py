import os
import re
import json
import pickle
from collections import OrderedDict

from conllu import parse


class DataLoader:
    def __init__(self):
        self.morpheme_labels_stoplist = {
            '1SG??', '1SG???', '1SGОЖИДАЛСЯ', '2SG?', '3SGПОЧЕМУ.НЕ.RFL?', '?', '???', '???ФОРМА', 'ABL?', 'ACC??',
            'ACCIN?', 'ADJ?', 'ADVZ???', 'DATLOC(?)', 'EVERY(TIKIN?)' 'FOC?',  'INDEF(?)', 'INDPS?', 'INSTR???',
            'INTS???', 'LOCALLНЕПОНЯТНО.ПОЧЕМУ', 'MISPRINT?', 'PS1SGГДЕ_ПАДЕЖ?'
        }

    def load_conllu(self, filename):
        """
        Загрузка файла в формате conllu и его парсинг.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read()
        parse_result = parse(data)
        filtered_result = []
        for i, el in enumerate(parse_result.copy()):
            if len(el) != 0:
                filtered_result.append(el)
        return filtered_result

    def write_conllu(self, filename, object):
        with open(filename, 'w') as f:
            for sent in object:
                f.write(sent.serialize())

    def load_non_labeled(self, filename):
        """
        Загрузка неразмеченной выборки.
        Преобразование в формат, аналогичный распаршенному conllu: списки словарей с ключами 'id' и 'form'.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read()
            sents_list = data.split('\n\n')
        words_list = [list(filter(None, el.split('\n'))) for el in sents_list]
        result_list = [[word.split('\t') for word in sent] for sent in words_list]
        ordered = [[OrderedDict(zip(['id', 'form'], word)) for word in sent] for sent in result_list]
        return ordered

    def extract_morphemes(self, morphemes):
        morphemes = morphemes.split()
        for morph in morphemes:
            if '_' in morph:
                try:
                    m, label = morph.split('_')
                except ValueError:
                    m, label = morph.split('_', 1)
                if '?' in label or len(label.split('_')) > 1 or re.search('[А-Яа-яЁё]', label):
                    self.morpheme_labels_stoplist.add(label)
                if label in self.morpheme_labels_stoplist:
                    label = 'X'
            else:
                m = morph
                label = 'ROOT'
            yield {'morpheme': m,
                   'label': label}

    def load_morphemes(self, morphemes_path):
        with open(morphemes_path) as f:
            data = f.read()
            sentences = list(filter(None, data.split('\n\n')))
        for sent in sentences:
            words = list(filter(None, sent.split('\n')))
            sent_words_with_morphemes = []
            for i, word in enumerate(words):
                form_morphemes = word.strip().split('\t')
                form, morphemes = form_morphemes[0], form_morphemes[1]
                morphemes = list(self.extract_morphemes(morphemes))
                sent_words_with_morphemes.append({'form': form, 'morphemes': list(morphemes)})
            yield sent_words_with_morphemes

    def load_json(self, path_to_json):
        with open(path_to_json) as outfile:
            return json.load(outfile, object_hook=OrderedDict)

    def save_json(self, path_to_save, object_to_save):
        with open(path_to_save, 'w') as outfile:
            outfile.write(
                json.dumps(object_to_save, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': '))
            )

    def load_model(self, model_path):
        """
        Загрузка сохранённых моделей.
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def pickle_model(self, lang, task, model):
        if not os.path.exists('models/{}'.format(lang)):
            os.mkdir('models/{}'.format(lang))
        with open('models/{}/{}_{}.pkl'.format(lang, lang, task), 'wb') as f:
            pickle.dump(model, f)


