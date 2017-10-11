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

		
	# print(qnID_dict['175648'])





if __name__ == '__main__':
	
	#Read raw train data
	data_dict = read_data('train_data.csv')
	qnID_dict = make_qnID_dict(data_dict)
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.1
	param['max_depth'] = 6
	param['silent'] = 1
	param['nthread'] = 4
	param['num_class'] = 4
	
	num_round = 5
	bst = xgb
	bstFin = xgb
	
	# print(len(qnID_dict.items()))
	i = 1
	for key in qnID_dict:

		features = []
		labels = []
		# print (key)
		for row in qnID_dict[key]:

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
			labels.append(int(row['Label']) - 1)
		X_train, X_test, y_train, y_test = train_test_split(
		features, labels, test_size=0.20, random_state=42)
		dtrain = xgb.DMatrix(X_train, label = y_train)
		dtest = xgb.DMatrix(X_test, label = y_test)
		watchlist = [(dtrain, 'train'), (dtest, 'test')]
		bst = xgb.train(param, dtrain, num_round, watchlist)
		bst.save_model('model1Prob.model')
		if(i == 2 ):
			bstFin = xgb.train(param, dtrain, num_round, watchlist, xgb_model = bst)
			bstFin.save_model('model2Prob.model')
			print(bstFin.predict(dtest))# bst.update(dtrain, i)
		elif (i > 2):
			bstFin = xgb.train(param, dtrain, num_round, watchlist, xgb_model = bstFin)
			bstFin.save_model('model3Prob.model')
		i += 1
	bstFin.save_model('xgb_update_prob.model')




		# bst = xgb.Booster({'nthread':4})
		# bst.load_model("xgb_pathClass.model")
	# dtest = xgb.DMatrix(features)
	# ypred = bstFin.predict(dtest)
	# print(ypred)

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






