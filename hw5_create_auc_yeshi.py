import pandas
import os
import sys
import re
from os import path
from sklearn.metrics import roc_curve, auc


DIR_PATH = os.path.normpath("./" + sys.argv[1])
TRUTH_FILE = 'testTruth.txt'

files_list = [x for x in os.listdir(DIR_PATH) if path.isfile(DIR_PATH+os.sep+x)]

ignoreFiles = ['.DS_Store', 'results.csv', 'hw5_create_auc.py', 'testTruth.txt'] # add files here to remove 
for fileToRemove in ignoreFiles:
	try:
		files_list.remove(fileToRemove)
	except:
		print('OK')

# truth = pandas.read_csv(TRUTH_FILE, delimiter='\t', names=['truth'], header=None)
#
# print(truth)

results = []
for filename in files_list:
	roc_auc_scores = {}
	try:
         print(filename)        
         filepath = os.path.join(DIR_PATH, filename)
         print(filepath)
         data = pandas.read_csv(filepath, delimiter='\t', names=['col1', 'col2', 'col3'], header=None)
         roc_auc_scores['filename'] = re.sub('^.*[0-9]{5,}_[0-9]{5,}_',"",filename)
         #print("Result1 is [", re.sub('^.*[0-9]{5,}\_[0-9]{5,}_' ,'',filename) )
         #print("Result2 is [", re.sub('[0-9]{5,}' ,'',filename ))
         #print('Filename ',filename)
    
         names=['col1', 'col2', 'col3']        
         for i in range(1,4):
             print("names ", names[i-1])
             try:
                 false_positive_rate, recall, thresholds = roc_curve(truth, data[names[i-1]], pos_label=i)
                 roc_auc_scores[str(i)] = auc(false_positive_rate, recall)
             except:
                 roc_auc_scores[str(i)] = 'error'

	except:
		roc_auc_scores['notes'] = 'error reading file'
		print('Error reading file ' + filename)

	results.append(roc_auc_scores)
	print('AUC scores for ' + filename + ' are %s %s %s' % 
		(roc_auc_scores['1'], roc_auc_scores['2'], roc_auc_scores['3']))

df = pandas.DataFrame(results)

cols = ['filename', '1', '2', '3']
df = df[cols]
df.to_csv(os.path.join(DIR_PATH, 'newResults.csv'))