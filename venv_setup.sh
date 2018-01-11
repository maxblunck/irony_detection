#!/bin/bash

virtualenv soprobroenv
source soprobroenv/bin/activate
pip install --upgrade pip
pip install nltk
pip install numpy
pip install scipy
pip install sklearn
pip install requests
pip install textblob
pip install vaderSentiment
