import os
import time
import uuid
import logging
from collections import defaultdict

from utils.data_loader import DataLoader


def check_form_to_morpheme(train, morphemes: list):
    assert(len(train) == len(morphemes), 'Len pipeline {} != len morphemes {}'.format(len(train), len(morphemes)))
    for i, sent in enumerate(train):
        morph_sent = morphemes[i]
        for j, w in enumerate(sent):
            assert(w['form'] == morph_sent[j]['form'])
    return True

def init_logging(file_name):
    fmt = logging.Formatter('%(asctime)-15s %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir_name = 'logs'
    log_file_name = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8] + '_%s.txt' % (file_name,)
    logging.info('Logging to {}'.format(log_file_name))
    logfile = logging.FileHandler(os.path.join(log_dir_name, log_file_name), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    return log_dir_name

def replace_morphemes(train, morphemes):
    print('Len train:', len(train))
    print('Len morphemes', len(morphemes))
    for i, sent in enumerate(train):
        morph_sent = morphemes[i].copy()
        if len(morph_sent) != len(sent):
            hyphen_count = sum([1 for w in sent if '-' in str(w['id'])])
            counter = 0
            while counter < hyphen_count:
                morph_sent = morphemes[i]
                for j, w in enumerate(sent):

                    if '-' in str(w['id']):
                        morph = [(index, m) for index, m in enumerate(morph_sent) if m['form'] == w['form']][0]
                        index_morph, morph = morph[0], morph[1]
                        if w['id'][2] - w['id'][0] == 1:
                            part1 = sent[j + 1]['form']
                            part2 = sent[j + 2]['form']
                            morph1 = morph['morphemes'][:-1]
                            morph2 = [morph['morphemes'][-1]]
                            morphemes[i].pop(index_morph)
                            morphemes[i].insert(index_morph, {'form': part1, 'morphemes': morph1})
                            morphemes[i].insert(index_morph + 1, {'form': part2, 'morphemes': morph2})
                            sent.pop(j)
                        if w['id'][2] - w['id'][0] == 2:
                            part1 = sent[j + 1]['form']
                            part2 = sent[j + 2]['form']
                            part3 = sent[j + 3]['form']
                            morph1 = morph['morphemes'][:-2]
                            morph2 = [morph['morphemes'][-2]]
                            morph3 = [morph['morphemes'][-1]]
                            morphemes[i].pop(index_morph)
                            morphemes[i].insert(index_morph, {'form': part1, 'morphemes': morph1})
                            morphemes[i].insert(index_morph + 1, {'form': part2, 'morphemes': morph2})
                            morphemes[i].insert(index_morph + 2, {'form': part3, 'morphemes': morph3})
                            sent.pop(j)
                        break
                    else:
                        continue
                counter += 1
                continue
            assert(len(morphemes[i]) == len(sent))
            for ind, word in enumerate(sent):
                assert(word['form'] == morphemes[i][ind]['form'])
            train.pop(i)
            train.insert(i, sent)
    return train, morphemes

def invert_dict(dictionary):
    new_dic = {}
    for k, v in dictionary.items():
        for x in v:
            new_dic.setdefault(x, []).append(k)
    return {k: sorted(v) for k, v in new_dic.items()}

def get_categories(lang_prefix, train):
    """
    Получение списка грамматических категорий из train и словаря соответствий частей речи и грамматических категорий.
    :param train:
    :return:
    """
    data_loader = DataLoader()
    unique_categories = set()
    pos2categories = defaultdict(set)
    for sent in train:
        for word in sent:
            if word['feats']:
                for feat in word['feats']:
                    unique_categories.add(feat)
                    pos2categories[word['upostag']].add(feat)
    categories = sorted(list(unique_categories))
    pos2categories = {k: sorted(list(v)) for k, v in pos2categories.items()}
    categories2pos = invert_dict(pos2categories)
    data_loader.save_json('data/grammar_data/{}_categories.json'.format(lang_prefix), categories)
    data_loader.save_json('data/grammar_data/{}_pos2categories.json'.format(lang_prefix),
                               pos2categories)
    return {'categories': categories,
            'pos2categories': pos2categories,
            'categories2pos': categories2pos}
