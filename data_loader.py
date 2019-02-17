from collections import OrderedDict

from conllu import parse


class DataLoader:
    def __init__(self):
        pass

    def load_conllu(self, filename):
        """
        Загрузка файла в формате conllu и его парсинг.
        Для парсинга файлов с менее чем 10 колонками использовался этот парсер:
        https://github.com/svetlana21/conllu
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read()
        return parse(data)

    def load_non_labeled(self, filename):
        """
        Загрузка неразмеченной выборки.
        Преобразование в формат, аналогичный распаршенному conllu: списки словарей с ключами 'id' и 'form'.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read()
            sents_list = data.split('\n\n')
        words_list = [el.split('\n') for el in sents_list]
        result_list = [[word.split('\t') for word in sent] for sent in words_list]
        if result_list[-1][-1] == ['']:
            result_list[-1].pop()
        ordered = [[OrderedDict(zip(['id', 'form'], word)) for word in sent] for sent in result_list]
        return ordered

