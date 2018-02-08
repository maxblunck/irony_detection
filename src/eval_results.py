import csv
from operator import itemgetter
import sys

"""
Returns the best evaluation scores for a "cv_results_*_*_*.csv" file
Needs the file name as command line argument
"""

def replace(name):
	if name == "DecisionTreeClassifier":
		return "Tree"
	elif name == "LogisticRegression":
		return "LogR."
	else:
		return name

if __name__ == '__main__':
		
	file = open(sys.argv[1], "r")

	reader = csv.reader(file)

	results = [x for x in reader][1:]

	sorted_by_acc = sorted(results, key=itemgetter(2), reverse=True)
	sorted_by_prec = sorted(results, key=itemgetter(3), reverse=True)
	sorted_by_rec = sorted(results, key=itemgetter(4), reverse=True)
	sorted_by_f1 = sorted(results, key=itemgetter(5), reverse=True)


	print("Top 5 Accuracy:")
	for i in range(5):
		print("{}\t{}\t{}".format(sorted_by_acc[i][2], replace(sorted_by_acc[i][0]), sorted_by_acc[i][1]))

	print("\nTop 5 Precision:")
	for i in range(5):
		print("{}\t{}\t{}".format(sorted_by_prec[i][3], replace(sorted_by_prec[i][0]), sorted_by_prec[i][1]))

	print("\nTop 5 Recall:")
	for i in range(5):
		print("{}\t{}\t{}".format(sorted_by_rec[i][4], replace(sorted_by_rec[i][0]), sorted_by_rec[i][1]))

	print("\nTop 10 F1:")
	for i in range(10):
		print("{}\t{}\t{}".format(sorted_by_f1[i][5], replace(sorted_by_f1[i][0]), sorted_by_f1[i][1]))
		features = sorted_by_f1[i][1].strip("[]").strip().split(",")


	print("\nTop F1-scores without f7:")
	for i in range(200):
		if "f7" not in sorted_by_f1[i][1] and "f4" not in sorted_by_f1[i][1]:
			print("{}\t{}\t{}".format(sorted_by_f1[i][5], replace(sorted_by_f1[i][0]), sorted_by_f1[i][1]))


