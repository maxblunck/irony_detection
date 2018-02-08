# Irony Detection In Amazon Reviews

## About

This app was developed and used for our software project (WS 17/18). 

The main purpose of the program is to use a machine learning approach to detect irony in customer reviews. When running the main program, several classifiers are trained and evaluated.

We use Elena Filatova's corpus containing ironic and non-ironic customer reviews from Amazon.com as our [data](https://github.com/ef2020/SarcasmAmazonReviewsCorpus/wiki).

## Setup 

We suggest running the `setup.sh` file. This creates a virtual python environment and installs  all dependencies of the app.

	$ bash setup.sh

Alternatively, you can manually install the following requirements:

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

 If not already activated, activate the virtualenv

	$ source sopro_env/bin/activate

To run the main program run `main.py`.

	$ cd src/
	$ python3 main.py

With the default settings, several classifiers will be trained on 80% of the data and tested on the other 20%. Results will be then printed out and also saved to the `results/` directory. In this setting, a certain feature-combination is used, which generated the best scores in prior experiments.

Changes can be made in `config.py`. Examples:

To generate cross-validation scores which can be compared to [Buschmeier et al.](http://acl2014.org/acl2014/W14-26/pdf/W14-2608.pdf), change the following variables to:

	split_ratio = 1.0
	validate = True

To choose a different combination of Features, modify the following variable:

	feature_selection = ['f1', 'f4', 'f7']

If you'd like to run the program for all possible combinations of the selected features, change the following variable to:

	use_all_variants = True

Feature specific options like the n-parameter of the bag-of-n-grams feature can also be adjusted. Changing the following variable as shown will make the feature extract uni- and bigrams:

	n_range_words = (1,2) 


See `config.py` itself for further options.

## App Structure

### Main Program
	- main.py 							> entry point to app, calls machine_learning.py's run()-function

### Feature Related Files
	- feature.py 						> provides an abstract Feature class
		|- ngram_feature.py 			> inherites from Feature, offers method for extracting F1 feature
			|- surface_patterns.py 		> inherites from NGramFeature, offers method for extracting F3 feature
		|- pos_feature.py 				> inherites from Feature, offers method for extracting F2 feature
		|- sent_rating_feature.py 		> inherites from Feature, offers method for extracting F4 feature
		|- punctuation_feature.py 		> inherites from Feature, offers method for extracting F5 feature
		|- contrast_feature.py 			> inherites from Feature, offers method for extracting F6 feature
		|- stars_feature.py 			> inherites from Feature, offers method for extracting F7 feature
	- feature_extraction.py 			> provides functions for extracting and concatenating feature vectors

### Machine Learning
	- machine_learning.py 				> includes run-function, which incorperates all ML related steps (training,testing,..)

### Other
	- corpus.py 						> contains a reading function to load corpus, can also be run to convert raw corpus
	- utilities.py						> collection of functions & helpers used throughout the app
	- config.py 						> file for adjusting setting and options

### Directories
	- src/								> holds all the source code above
	- results/ 							> default location where test/validation results are saved
	- corpus/ 							> contains complete corpus in a single csv-file (shuffled)			

