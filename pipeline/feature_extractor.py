from collections import OrderedDict

import numpy as np
from nltk.util import ngrams


class FeatureExtractor:
    """
    Класс для извлечения признаков.
    """

    def __init__(self, features=None):
        self.features = features
        self.hyphen_parts_indexes = OrderedDict()
        self.morpheme_preproc = None
        self.morphemes = None

    def set_morpheme_preproc(self, morpheme_preproc):
        self.morpheme_preproc = morpheme_preproc
        self.morphemes = self.morpheme_preproc.morphemes

    def set_morphemes_fold(self, morphemes_fold):
        self.morphemes = morphemes_fold

    def del_hyphen_parts(self, dataset):
        """
        Удаление из train'а токенов дефисных написаний (с id-шниками типа 1-2 и без тегов)
        """
        for i, sent in enumerate(dataset.copy()):
            for j, word in enumerate(sent):
                if type(word['id']) == tuple:
                    self.hyphen_parts_indexes[i] = {'pre_hyphen': j + 1,
                                                    'post_hyphen': j + 2}
                    dataset[i].pop(j)
        return dataset

    def all_words_features(self, sent, i, sent_id):
        """
        Получение признаков, которые нужно извлечь для любого слова, вне зависимости от его положения в окне.
        Это такие признаки, как:
            1) токен в uppercase (или нет);
            2) токен с большой буквы (или нет);
            3) токен - это цифра (или нет);
            4) первая и последняя буквы;
            5) если длина слова > 1, то префиксы и суффиксы длины от 2 до 4 символов.
        """
        word = sent[i]['form']
        word_features = [word.isupper(), word.istitle(), word.isdigit(), word[0], word[-1]]  # признаки 1-4
        if len(word) > 1:                                       # префиксы и суффиксы в зависимости от длины слова
            word_features.extend([word[:2], word[-2:]])
        if len(word) > 2:
            word_features.extend([word[:3], word[-3:]])
        if len(word) > 3:
            word_features.extend([word[:4], word[-4:]])
        return word_features

    def make_right_context_features(self, sent, i, sent_id):
        """
        Функция, формирующая признаки правого контекста текущего токена.
        Это те признаки, которые извлекаются с помощью функции all_words_features.
        Функция также формирует список слов в правом контексте.
        В дальнейшем это требуется для формирования списка всех слов окна, который передаётся в функцию ngrams.
        """
        word1 = sent[i + 1]['form']
        word1_feat = self.all_words_features(sent, i + 1, sent_id)
        r_context = [word1]
        r_context_features = dict(zip(['+1:word_is_upper', '+1:word_is_title', '+1:word_is_digit',
                                       '+1:pref[0]', '+1:suf[-1]',
                                       '+1:pref[:2]', '+1:suf[-2:]', '+1:pref[:3]', '+1:suf[-3:]',
                                       '+1:pref[:4]', '+1:suf[-4:]'], word1_feat))
        if i == len(sent) - 3:
            word2 = sent[i + 2]['form']
            word2_feat = (self.all_words_features(sent, i + 2, sent_id))
            r_context.append(word2)
            w2_features = dict(zip(['+2:word_is_upper', '+2:word_is_title', '+2:word_is_digit',
                                    '+2:pref[0]', '+2:suf[-1]',
                                    '+2:pref[:2]', '+2:suf[-2:]', '+2:pref[:3]', '+2:suf[-3:]',
                                    '+2:pref[:4]', '+2:suf[-4:]'], word2_feat))
            r_context_features.update(w2_features)
        if i < len(sent) - 3:
            word2 = sent[i + 2]['form']
            word2_feat = (self.all_words_features(sent, i + 2, sent_id))
            word3 = sent[i + 3]['form']
            word3_feat = (self.all_words_features(sent, i + 3, sent_id))
            r_context.extend([word2, word3])
            w2_features = dict(zip(['+2:word_is_upper', '+2:word_is_title', '+2:word_is_digit',
                                    '+2:pref[0]', '+2:suf[-1]',
                                    '+2:pref[:2]', '+2:suf[-2:]', '+2:pref[:3]', '+2:suf[-3:]',
                                    '+2:pref[:4]', '+2:suf[-4:]'], word2_feat))
            w3_features = dict(zip(['+3:word_is_upper', '+3:word_is_title', '+3:word_is_digit',
                                    '+3:pref[0]', '+3:suf[-1]',
                                    '+3:pref[:2]', '+3:suf[-2:]', '+3:pref[:3]', '+3:suf[-3:]',
                                    '+3:pref[:4]', '+3:suf[-4:]'], word3_feat))
            r_context_features.update(w2_features)
            r_context_features.update(w3_features)
        return r_context_features, r_context

    def make_left_context_features(self, sent, i, sent_id):
        """
        То же, что make_right_context_features, только для левого контекста.
        """
        word1 = sent[i - 1]['form']
        word1_feat = self.all_words_features(sent, i - 1, sent_id)
        l_context = [word1]
        l_context_features = dict(
            zip(['-1:word_is_upper', '-1:word_is_title', '-1:word_is_digit', '-1:pref[0]', '-1:suf[-1]',
                 '-1:pref[:2]', '-1:suf[-2:]', '-1:pref[:3]', '-1:suf[-3:]', '-1:pref[:4]', '-1:suf[-4:]'], word1_feat))
        if i == 2:
            word2 = sent[i - 2]['form']
            word2_feat = self.all_words_features(sent, i - 2, sent_id)
            l_context.insert(0, word2)
            w2_features = dict(
                zip(['-2:word_is_upper', '-2:word_is_title', '-2:word_is_digit', '-2:pref[0]', '-2:suf[-1]',
                     '-2:pref[:2]', '-2:suf[-2:]', '-2:pref[:3]', '-2:suf[-3:]', '-2:pref[:4]', '-2:suf[-4:]'],
                    word2_feat))
            l_context_features.update(w2_features)
        if i > 2:
            word2 = sent[i - 2]['form']
            word2_feat = self.all_words_features(sent, i - 2, sent_id)
            word3 = sent[i - 3]['form']
            word3_feat = self.all_words_features(sent, i - 3, sent_id)
            l_context.insert(0, word2)
            l_context.insert(0, word3)
            w2_features = dict(
                zip(['-2:word_is_upper', '-2:word_is_title', '-2:word_is_digit', '-2:pref[0]', '-2:suf[-1]',
                     '-2:pref[:2]', '-2:suf[-2:]', '-2:pref[:3]', '-2:suf[-3:]', '-2:pref[:4]', '-2:suf[-4:]'],
                    word2_feat))
            w3_features = dict(
                zip(['-3:word_is_upper', '-3:word_is_title', '-3:word_is_digit', '-3:pref[0]', '-3:suf[-1]',
                     '-3:pref[:2]', '-3:suf[-2:]', '-3:pref[:3]', '-3:suf[-3:]', '-3:pref[:4]', '-3:suf[-4:]'],
                    word3_feat))
            l_context_features.update(w2_features)
            l_context_features.update(w3_features)
        return l_context_features, l_context

    def ngrams(self, window):
        """
        Признаки-биграммы и признаки-триграммы.
        """
        ngrams_2 = list(ngrams(window, 2))
        ngrams_3 = list(ngrams(window, 3))
        bigrams = ['{} {}'.format(ngrams_2[i][0], ngrams_2[i][1]) for i in range(len(ngrams_2))]
        trigrams = ['{} {} {}'.format(ngrams_3[i][0], ngrams_3[i][1], ngrams_3[i][2]) for i in range(len(ngrams_3))]
        bigr_features = dict(zip(['bi_1', 'bi_2', 'bi_3', 'bi_4', 'bi_5', 'bi_6'], bigrams))
        trigr_features = dict(zip(['tri_1', 'tri_2', 'tri_3', 'tri_4', 'tri_5'], trigrams))
        return bigr_features, trigr_features

    def empty_morphemes(self):
        empty = np.zeros_like(self.morpheme_preproc.labels, dtype=int)
        empty = empty.tolist()
        return dict(zip(self.morpheme_preproc.label2ind.keys(), empty))

    def morpheme_features(self, sent, i, sent_id):
        word = sent[i]['form']
        if i < len(self.morphemes[sent_id]):
            word_with_morphemes = self.morphemes[sent_id][i]
            if word != word_with_morphemes['form']:
                # print('empty')
                # print(sent)
                # print(self.morphemes[sent_id])
                return self.empty_morphemes()
            return dict(zip(self.morpheme_preproc.label2ind.keys(), self.morpheme_preproc.one_hot(word_with_morphemes)))
        else:
            # print('empty')
            # print(sent)
            # print(self.morphemes[sent_id])
            # assert 0 == 1

            return self.empty_morphemes()

    def word2features(self, sent, i, sent_id, add_postags=False, ngrams=False):
        """
        Функция, формирующая полный список признаков:
            1) токен в uppercase (или нет);
            2) токен с большой буквы (или нет);
            3) токен - это цифра (или нет);
            4) первая и последняя буквы;
            5) если длина слова > 1, то префиксы и суффиксы длины от 2 до 4 символов;
            6) всё вышеперечисленное для правого контекста;
            7) всё вышеперечисленное для левого контекста;
            8) сам токен в lowercase;
            9) bias = 1.0;
            10) если токен является началом предложения, 'BOS' = True;
            11) если токен является началом предложения, 'EOS' = True;
            12) биграммы;
            13) триграммы.
        Если значение add_postags = True, то в качестве признака добавляется postag
        (для обучения классификаторов, предсказывающих грамматические категории).
        """
        word = sent[i]['form']
        word_feat = self.all_words_features(sent, i, sent_id)
        features = dict(zip(['word_is_upper', 'word_is_title', 'word_is_digit', 'pref[0]', 'suf[-1]',
                             'pref[:2]', 'suf[-2:]', 'pref[:3]', 'suf[-3:]', 'pref[:4]', 'suf[-4:]'], word_feat))

        features.update({'word': word.lower(),
                         'bias': 1.0})
        if len(sent) > 1:
            if i == 0:
                features['BOS'] = True
                right_context = self.make_right_context_features(sent, i, sent_id)
                features.update(right_context[0])
                window = right_context[1]
                window.insert(0, word)
            elif i == len(sent) - 1:
                features['EOS'] = True
                left_context = self.make_left_context_features(sent, i, sent_id)
                features.update(left_context[0])
                window = left_context[1]
                window.append(word)
            else:
                left_context = self.make_left_context_features(sent, i, sent_id)
                features.update(left_context[0])
                right_context = self.make_right_context_features(sent, i, sent_id)
                features.update(right_context[0])
                window = left_context[1]
                window.append(word)
                window.extend(right_context[1])
            if ngrams:
                ngrams = self.ngrams(window)
                features.update(ngrams[0])
                features.update(ngrams[1])
        else:
            features['BOS'] = True
        if self.morphemes:
            features.update(self.morpheme_features(sent, i, sent_id))
        if add_postags == True:
            features.update({'postag': sent[i]['upostag']})
        return features

    def sent2features(self, sent, sent_id, postags=False, ngrams=False):
        """
        Все признаки для одного предложения.
        """
        return [self.word2features(sent, i, sent_id, postags, ngrams) for i in range(len(sent))]

    def sent2features_gc(self, sent, sent_id, category, pos_tags):
        """
        Все признаки для одного предложения.
        """
        sent_features = []
        sent_labels = []
        for i in range(len(sent)):
            if sent[i]['upostag'] in ['NOUN', 'VERB', 'ADJ', 'PRON', 'ADP', 'AUX', 'ADV']:
                if sent[i]['feats']:
                    if sent[i]['upostag'] in pos_tags:
                        sent_features.append(self.word2features(sent, i, sent_id, add_postags=True))
                        sent_labels.append(self.word2label_gc(sent[i], category))
            else:
                if sent[i]['upostag'] in pos_tags:
                    sent_features.append(self.word2features(sent, i, sent_id, add_postags=True))
                    sent_labels.append(self.word2label_gc(sent[i], category))
        assert(len(sent_features) == len(sent_labels))
        return sent_features, sent_labels

    def word2label_gc(self, word, category):
        """
        Классы для одного слова, если классы определяются по грамматической категории.
        На вход подаётся слово и интересующая ГК.
        Если слово соответствует переданной ГК, то метка класса определяется по её грам. значению, иначе - О.
        """
        if word['feats'] is not None and category in word['feats']:
            label = word['feats'][category]
        else:
            label = 'O'
        return label

    def sent2labels(self, sent, category, pos=False):
        """
        Все классы для одного предложения.
        """
        if pos == True:
            sent_labels = [sent[i]['upostag'] if sent[i]['upostag'] != 'PROPN' else 'NOUN' for i in range(len(sent))]
            sent_labels = [l if l != '_' else 'X' for l in sent_labels ]
        else:
            sent_labels = [self.word2label_gc(sent[i], category) for i in range(len(sent))]
        return sent_labels

    def add_pos_features(self, X, y_pred):
        """
        Добавление уже предсказанных частеречных тегов в качестве признаков.
        """
        pos_labels = y_pred.copy()
        for sent_ind in range(len(X)):
            sent_labels = pos_labels.pop(0)
            for word_ind in range(len(X[sent_ind])):
                X[sent_ind][word_ind].update({'postag': sent_labels[word_ind]})
        return X
