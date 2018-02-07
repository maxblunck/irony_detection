class Feature:
	"""Class representing an abstract feature
	
	extract():
		- needs to be overwritten by subclasses
		- should take a corpus instance (dict) as an arg
		- should return np.array containing feature values

	get_feature_names():
		- needs to be overwritten by subclasses
		- should return a list of feature descriptions
		  corresponding to feature vector
	"""

	def extract():
		raise NotImplementedError

	def get_feature_names():
		raise NotImplementedError
