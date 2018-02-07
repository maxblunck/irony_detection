'''
Configuration file
'''

'''
Data split ratio:
- Set to 1.0 to replicate comparison-paper scores (and set validate below)
- When set to 1.0, be sure to turn off testing
'''
split_ratio = 0.8


''' 
Choose mode:
- Set validate to True here, to replicate comparison-paper scores (10-fold cv)
- If test is set to True, training will also be activated
'''
validate = False
tune = False # takes ~24h
train = True
test = True


'''
Choose path and files to output results
- Each will be ovewritten if unchanged to previous run!
'''
test_result_out = "../results/test_results.csv"
validation_result_out = "../results/cv_results.csv"
misclassification_out = "../results/misclassifications.txt"


'''
Evaluation Utilities:
- plot_weights: visualize top weights (of SVM)
- log_misclassification: output all missclassfied instances (by SVM)
- Setting one of the below to True will also activate training
'''
plot_weights = False
log_misclassifications = True


'''
Select features to be used. Array must contain at least one feature:
f1 : n-gram feature
f2 : pos feature
f3 : surface-patterns feature
f4 : sent/rating-contrast feature
f5 : punctuation feature
f6 : contrast feature
f7 : number of stars (rating)
'''
#feature_selection = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
feature_selection = ['f1', 'f3', 'f4', 'f7']


'''
Setting to True will loop through & validate/train/test all possible combinations
of the features given in feature_selection
'''
use_all_variants = False


'''
Feature Options:
- Choose range of n for bag-of-X features. 
	- (1,1): only unigrams; 
	- (1,2): uni- + bi-grams
- Choose high-freq. threshold for surface-patterns feature
'''
n_range_words = (2,2) 
n_range_surface_patterns = (1,1)
n_range_lemmas = (1,1)
sp_threshold = 1000


'''
Print extra stats like vocabulary sizes
'''
print_stats = True


'''
Select csv-file containing the corpus (as created with corpus.py)
'''
corpus_path = "../corpus/corpus_shuffled.csv"

