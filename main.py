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
	

# to get all the sentences picked from the KB to the same coulmn
def clean_data(raw_train_data):
	clean_train_data = []
	
	for row in raw_train_data:
		del row[13:17]
		clean_train_data.append(row)

	for row in clean_train_data:
		for i in range(len(row)):
			if(i > 12 and i < 19):
				try:
					number = float(row[i])
					del row[i]
				except ValueError:
					print()
				if(len(row[i].split()) < 5 or '>' in row[i]):
					del row[i]

	return clean_train_data
		# try:
		# 	number = float(row[15])
		# 	del row[15]
		# except ValueError:
		# 	print("not a number")
		# try:
		# 	number = float(row[15])
		# 	del row[15]
		# except ValueError:
			# print("not a number")
		# try:
		# 	number = float(row[13])
		# 	del row[13]
		# except ValueError:
		# 	print("not a number")
		# try:
		# 	number = float(row[13])
		# 	del row[13]
		# except ValueError:
		# 	print("not a number")

def write_data(clean_train_data):
	with open('clean_data.csv','w') as fin:
		writer = csv.writer(fin, quoting=csv.QUOTE_ALL)
		writer.writerows(clean_train_data)

#remove stop words and return the list
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




if __name__ == '__main__':
	# question_keywords = []
	# path_kw = []
	# ans = []
	features = []
	labels = []
	#Read raw train data
	data_dict = read_data('clean_data.csv')
	for row in data_dict:

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
		path_feat[3] = VectorSim.greedy_matching(question_keywords, path_words + ans)

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

            
		# print(path_feat[7])

		# break
	X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.20, random_state=42)
	dtrain = xgb.DMatrix(X_train, label = y_train)
	dtest = xgb.DMatrix(X_test, label = y_test)
	# test_label = labels[-10:]
	param = {}
	param['objective'] = 'multi:softmax'
	param['eta'] = 0.1
	param['max_depth'] = 6
	param['silent'] = 1
	param['nthread'] = 4
	param['num_class'] = 4
	watchlist = [(dtrain, 'train'), (dtest, 'test')]
	num_round = 5
	bst = xgb.train(param, dtrain, num_round, watchlist)
	# bst.update(dtrain, 10)
	y_pred = bst.predict(dtest)
	count = 0
	bst.save_model('xgb_pathClass.model')
	for i in range(len(y_test)):
		if(y_test[i] != y_pred[i]):
			count = count + 1
	# print('pred_prob: ',y_pred)
	# print('test_label: ',y_test)
	# pred_label = np.argmax(pred_prob)
	print("wrong prediction count: ", count)
	# error_rate = np.sum(pred_label != test_label) / test_label.shape[0]
	print('test error using softprob : ',count / len(y_test))
	print(classification_report(y_test, y_pred))




	
	
