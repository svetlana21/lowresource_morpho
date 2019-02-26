#!/usr/bin/env bash
set -e

python3 -m pipeline.inference --lang evn --save-to /home/svetlana/data/PycharmProjects/lowresource_morpho/test_data/annotated/new/evn.test.ud
python3 -m pipeline.inference --lang sel --save-to /home/svetlana/data/PycharmProjects/lowresource_morpho/test_data/annotated/new/sel.test.ud
python3 -m pipeline.inference --lang krl --save-to /home/svetlana/data/PycharmProjects/lowresource_morpho/test_data/annotated/new/krl.test.ud
python3 -m pipeline.inference --lang olo --save-to /home/svetlana/data/PycharmProjects/lowresource_morpho/test_data/annotated/new/olo.test.ud
python3 -m pipeline.inference --lang lud --save-to /home/svetlana/data/PycharmProjects/lowresource_morpho/test_data/annotated/new/lud.test.ud
python3 -m pipeline.inference --lang vep --save-to /home/svetlana/data/PycharmProjects/lowresource_morpho/test_data/annotated/new/vep.test.ud
