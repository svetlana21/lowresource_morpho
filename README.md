# lowresource_morpho
POS и морфо-теггер для малоресурсных языков (Dialog 2019)

0. Установить requirements.
1. Положить обучающие данные в папку data.
Ссылка на данные: https://lowresource-lang-eval.github.io/content/data/index_data.html
Если есть данные морфемной сегментации, положить обучающие датасеты в data/morpheme
2. Положить тестовые данные в test_data. Если есть тестовые выборки с готовой морфемной сегментацией, положить в test_data/morpheme

##Обучение

Запустить run_train.sh.

--lang - язык

--option {cv,train}:
1) train - учим модели на всём датасете
2) cv - кросс-валидация на 5 фолдов с выводом метрик качества

--morphemes - добавлять ли морфемные фичи (type=bool, default=False)

--categories - обучать ли модели для грамматических категорий (FEATS) (type=bool, default=True)

## Inference

Запустить скрипт run_inference.sh

--lang - язык

--save-to - путь к новому файлу с разметкой

--morphemes - добавлять ли морфемные фичи (type=bool, default=False)
