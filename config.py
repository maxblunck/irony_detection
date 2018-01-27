'''
Configuration file
'''

''' 
Choose mode
'''
validate = False
tune = False # takes ~24h
train = True
test = True


'''
Choose wheter input vectors should be extracted again.
For first time use of programm: Variable needs to be set to True.
If set to False: Pre-extracted vectors are loaded from pickle file.
'''
re_extract = True


'''
Choose file to output test results. Path relative to src code
'''
test_result_out = "test_results.csv"


'''
Select features to be used. Array must have at least one feature:
f1 : n-gram feature
f2 : pos feature
f3 : surface-patterns feature
f4 : sent/rating-contrast feature
f5 : punctuation feature
f6 : pos/neg-phrase feature
f7 : number of stars (rating)
f8 : lemma n-gram feature

best so far ['f1' ,'f4', 'f5', 'f7'] 
'''
feature_selection = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']


'''
Setting to True will loop through and train/test all possible combinations
of the features entered in feature_selection.
'''
use_all_variants = True


'''
Choose range of n for bag-of-X features. 
(1,1): only unigrams; 
(1,2): uni- + bi-grams
until fixed: n1==n2!
'''
n_range_words = (1,1) 
n_range_surface_patterns = (1,1)
n_range_lemmas = (1,1)
