'''
Configuration file
'''

''' 
Choose mode
'''
validate = True
tune = False
train = False
test = False


'''
Choose wheter input vectors should be extracted again.
For first time use of programm: Variable needs to be set to True.
If set to False: Pre-extracted vectors are loaded from pickle file.
'''
re_extract = True


'''
Select features to be used. Array must have at least one feature:
f1 : n-gram feature
f2 : pos feature
f3 : surface-patterns feature
f4 : sent/rating-contrast feature
f5 : punctuation feature
f6 : pos/neg-phrase feature
'''
feature_selection = ['f1', 'f2', 'f3', 'f4'] 


'''
Choose range of n for bag-of-X features. 
(1,1): only unigrams; 
(1,2): uni- + bi-grams
until fixed: n1==n2!
'''
n_range_words = (1,1) 
n_range_surface_patterns = (1,1)