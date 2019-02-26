#!/usr/bin/env bash

set -e

python3 -m pipeline.train --lang evn --option train
python3 -m pipeline.train --lang sel --option train
python3 -m pipeline.train --lang krl --option train
python3 -m pipeline.train --lang lud --option train
python3 -m pipeline.train --lang olo --option train
python3 -m pipeline.train --lang vep --option train
