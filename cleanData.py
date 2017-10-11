import csv

with open ('uncleaned_data.csv', 'r') as fin:
	reader = csv.reader(fin)
	cleaned_data = []
	cleaned_data.append(next(reader))
	for row in reader:
		if ' '.join(row).strip():

			appendStr = ''
			for i in range(len(row)):
				if i >= 13 and i < 22 :
					# appendStr = ''
					if(row[i].find('->') == -1 and row[i] != 'N/A'):
						appendStr = appendStr + row[i]
					
						
					# try:
					# 	number = float(row[i])
					# 	del (row[i])
					# except ValueError:
					# 	print()
					
			row[13] = appendStr
			row[14] = row[26]
			row[15] = row[27]
			cleaned_data.append(row[:16])
	print(cleaned_data[4], cleaned_data[5])

with open('clean_data.csv','w') as fin:
	writer = csv.writer(fin, quoting=csv.QUOTE_ALL)
	writer.writerows(cleaned_data)




