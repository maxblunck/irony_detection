# Sarcasm Detection In Amazon Reviews

## About

This app was developed and used for our software project (WS 17/18). 

The main purpose of the programm is to use a machine learning approach to detect irony in customer reviews. When running the main programm, several classifiers are trained and evaluated.

We use Elena Filatova's corpus containing ironic and non-ironic customer reviews from Amazon.com as our [data](https://github.com/ef2020/SarcasmAmazonReviewsCorpus/wiki).

## Setup 

We suggest running the `setup.sh` file. This creates a virtual python environment and installs  all dependencies of the app.

	$ bash setup.sh

After running the setup, you will need to activate the virtualenv

	$ source sopro_env/bin/activate

Alternatively, you can manually install the following requirements.

## Requirements

The program requires NLTK, NumPy, SciPy, SciKit Learn, requests, textblob and matplotlib.
Please note that SciPy and NumPy need to be installed before SciKit Learn.

    $ pip install --upgrade pip
	$ pip install nltk
	$ pip install numpy
	$ pip install scipy
	$ pip install sklearn
	$ pip install requests
	$ pip install textblob
	$ python -mpip install -U matplotlib
	
## Run

To run the main programm run `main.py`

	$ cd src/
	$ python3 main.py

With the default settings, several classifiers will be trained on 80% of the data and tested on the other 20%. Results will be then printed out and also saved to the `results/` directory. In this setting, a certain feature-combination is used which generated the best scores in various experiments.

Changes can be made in `config.py`. 
To generate cross-validation scores which can be compared to [Buschmeier et al.](http://acl2014.org/acl2014/W14-26/pdf/W14-2608.pdf), change the following variables:

	split_ratio = 1.0
	validate = True

See `config.py` itself for further options.