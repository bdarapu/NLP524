'''
clean the questions data provided in qno_list and adds the question to a single file "cleaned_data_questions.csv"
'''
import csv

#convert txt to csv
def txt_csv(txt_file, csv_file):

	
	in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
	out_csv = csv.writer(open(csv_file, 'w'))
	out_csv.writerows(in_txt)

qno_list = []
with open ('qno.csv', 'r') as fin:
	reader = csv.reader(fin)
	for row in reader:
		qno_list.append(' '.join(row))

print(qno_list[:5])


#clean wuestion data 
def cleanQuestion(qno_list):
	cleaned_data = []
	header = []
	for q_no in qno_list:
		# print(q_no)
		in_txt = csv.reader(open("questions_paths/" + q_no, "r"), delimiter = '\t')
		csv_name = q_no[:q_no.find('.') + 1] + "csv"
		out_csv = csv.writer(open("questions_csv/" + csv_name, 'w'))
		out_csv.writerows(in_txt)
		with open ("questions_csv/" + csv_name, 'r') as fin:
			reader = csv.reader(fin)
			try:
				header = next(reader)
			except:
				continue
			for row in reader:
				if ' '.join(row).strip(): #Not empty
					# print("test:")
					appendStr = ''
					for i in range(len(row)):
						
						if i >= 13 and i < 22 :
							# appendStr = ''
							if(row[i].find('->') == -1 and row[i] != 'N/A'):
								appendStr = appendStr + row[i]
					try:		
						row[13] = appendStr
					except :
						print(row)
					# print(row[:14])
					cleaned_data.append(row[:14])
	cleaned_data.insert(0, header)
	return cleaned_data
			

if __name__ == '__main__':
	cleaned_data = cleanQuestion(qno_list)
	with open('cleaned_data_questions.csv','w') as fin:
		writer = csv.writer(fin, quoting=csv.QUOTE_ALL)
		writer.writerows(cleaned_data)
	print(cleaned_data[:1])
	
