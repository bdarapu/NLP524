'''
load xgboost model and predict classes on "cleaned_data_questions.csv"
'''

import csv 
from nltk.corpus import stopwords
import numpy as np
import VectorSim
import itertools
from sklearn.linear_model import LogisticRegression as lr
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from operator import itemgetter


#read data as a dictionary object and returns it
def read_data(data_info):
	data_dict = csv.DictReader(open(data_info, 'r'))
	return data_dict

def get_keywords(word_list):
	filtered_words = [word for word in word_list if word not in stopwords.words('english')]
	return ' '.join(filtered_words)

# read path - read upto '.' or end of string whichever comes first
def getIRSentence(data):
	data_str = ' '.join(data)
	if(data_str.find('.') != -1):
		return data_str[:data_str.find('.')].split(" ")
	else :
		return data_str.split(" ")
	# print(data_str[:data_str.find('.')])

def get_q_overlap_count(q, path):
    count = 0
    q_sent = ' '.join(q)
    for kw in path:
        if kw in q_sent:
            count += 1

    return count

def make_qnID_dict(data_dict):
	qnID_dict = {}
	for row in data_dict:
		if (row['QID']) in qnID_dict:
			qnID_dict[row['QID']].append(row)
		else:
			qnID_dict[row['QID']] = [row]
	return qnID_dict




if __name__ == '__main__':
	features = []
	# labels = []
	#Read raw train data
	data_dict = read_data('test5Questions.csv')
	qnID_dict = make_qnID_dict(data_dict)
	bst = xgb.Booster({'nthread':4})
	bst_prob = xgb.Booster({'nthread':4})
	bst.load_model("xgb_update.model")
	bst_prob.load_model("xgb_update_prob.model")
	ypred_dict = {}
	yprob_dict = {}
	for key in qnID_dict:
		# print (key)
		features = []
		labels = []
		for row in qnID_dict[key]:
			# print(row)

			# get keywords from question
			question_keywords = get_keywords(row['Question'].split(" "))
			# print(question_keywords)

			# get keywords from path
			path_words = get_keywords(getIRSentence(row['Path'].split(" ")))
			# print(path_words)

			# Answer
			ans = row[row['Answer']]
			# print(ans)

			# Path-Features
			path_feat = np.zeros(8)
			#questioin keywords and path keywords
			path_feat[0] = VectorSim.greedy_matching(question_keywords, path_words)

			#answer and path keywords
			path_feat[1] = VectorSim.greedy_matching(ans, path_words)

			# question keywords and answer
			path_feat[2] = VectorSim.greedy_matching(question_keywords, ans)
			
			# question keywords, answer keywords and path keywords
			path_feat[3] = VectorSim.greedy_matching(question_keywords, path_words + ans);

			#checking for path words and question keywords
			path_words = [p for p in path_words if p not in question_keywords]
			if (path_words == []):
				path_feat[4] = 1
				path_feat[5] = 1
			else:
				word_sim_scores = [VectorSim.greedy_matching(q, p) for q, p in itertools.product(question_keywords, path_words)]
				path_feat[5] = max(word_sim_scores)
				path_feat[6] = min(word_sim_scores)

			#question keywords, path words and answer overlap count
			path_feat[7] = get_q_overlap_count(question_keywords, path_words + [ans])
			features.append(path_feat)
			# labels.append(int(row['Label']) - 1)
		
		dtest = xgb.DMatrix(features)
		ypred = bst.predict(dtest)
		yprob = bst_prob.predict(dtest)
		print(ypred)
		print(yprob)
		ypred = ypred.tolist()
		if key in ypred_dict:
			print("if: ")
			ypred_dict[key].append(ypred)
		else:
			print("else: ")
			ypred_dict[key] = ypred
		if key in yprob_dict:
			print("if: ")
			yprob_dict[key].append(yprob)
		else:
			print("else: ")
			yprob_dict[key] = yprob

	print(len(ypred_dict.items()))
	print(len(ypred_dict.items()))

	


	# appending labels to the data 
	for key in ypred_dict:
		# print("key: ",key)
		l = ypred_dict[key]
		p = yprob_dict[key]
		for i in range(len(ypred_dict[key])):
			qnID_dict[key][i]['Label'] = l[i]
			# print("l[i]: ",l[i])
			# print("p: ",p[i][int(l[i])])
			qnID_dict[key][i]['Prob'] = p[i][int(l[i])]
			# print(qnID_dict[key][i])

	sortedqnID_dict = {}
	for key in qnID_dict:
		l = qnID_dict[key]
		newlist = sorted(l, key=lambda k: (-k['Label'], -k['Prob']))
		sortedqnID_dict[key] = newlist

	for key in sortedqnID_dict:
		l = sortedqnID_dict[key]
		with open('./LabelledPaths/orderedPaths_'+key+'.csv', 'w') as csvfile:
		    fieldnames = ['QID', 'Question', 'A', 'B', 'C',	'D', 'Answer', 'PathID', 'PathLen',	'PrScores',	'EdgeScores', 'AvgPrScore', 'AvgEdgeScore','Path','Label','Prob']
		    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		    writer.writeheader()
		    writer.writerows(l)
			


	# for key in ypred_dict:
	# 	print("key: ",key)
	# 	# i = 0
	# 	l = ypred_dict[key]
	# 	for i in range(len(ypred_dict[key])):
	# 		print(l[i])
	# 		# i += 1

	# for i in range(len(qnID_dict[key])):
	# 	# qnID_dict[key][i]['label'] = ypred_dict[key][i]
	# 	print(qnID_dict[key][i])


	# predicted_data = []
	# with open('cleaned_data_questions.csv','r') as fin:
	# 	reader = csv.reader(fin)
	# 	header = next(reader)
	# 	i = 0
	# 	for row in reader:
	# 		row.append(ypred[i])
	# 		i += 1
	# 		predicted_data.append(row)
	# 	predicted_data.insert(0, header)

	# with open ('cleaned_data_predictions.csv', 'w') as fin:
	# 	writer = csv.writer(fin, quoting=csv.QUOTE_ALL)
	# 	writer.writerows(predicted_data)







