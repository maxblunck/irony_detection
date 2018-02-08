#!/bin/bash

virtualenv sopro_env
source sopro_env/bin/activate

pip install --upgrade pip
pip install nltk
pip install numpy
pip install scipy
pip install sklearn
pip install requests
pip install textblob

python -mpip install -U matplotlib
