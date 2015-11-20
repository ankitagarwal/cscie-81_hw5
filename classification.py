from config import *

from collections import OrderedDict

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.linear_model.logistic import LogisticRegression

from sklearn.naive_bayes import GaussianNB

import csv
import numpy as np
import datetime
import concurrent.futures
from queue import Queue
import os


class classification:
    resultQueue = Queue()
    def load_training_data(self):
        return self.load_tab_file(TRAINING_DATA_FILE)

    def load_training_label(self):
        fin = open(TRAINING_LABEL_FILE, 'r')
        data = fin.read().split('\n')
        # Dropping last item as it is an empty object
        del data[-1]
        return data

    def load_test_data(self):
        return self.load_tab_file(TEST_DATA_FILE)

    def load_tab_file(self, file):
        data = []
        with open(file) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Dropping last item as it is an empty object
                data.append([None if x == 'NA' else float(x) for x in row[:-1]])
        return data

    # Remove missing values from data and replace then with NaN for imputer to work.
    def remove_missing_values(self, arr):
        X = arr
        for row_ind, row in enumerate(arr):
            for col_ind, element in enumerate(row):
                if (element == 'NA'):
                    X[row_ind][col_ind] = float('NaN') #Fixed number to identify missing values.
                    # X[row_ind][col_ind] = 2 #Ignore imputation for now, it takes forever.
        return X

    def create_binary_labels(self, classes, label):
        binary_labels = []
        for clas in classes:
            if int(clas) == int(label):
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        return binary_labels


    def create_binary_labels(self, classes, label):
        binary_labels = []
        for clas in classes:
            if int(clas) == int(label):
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        return binary_labels


    def train(self, X, y, test_data, classifier, key):
        global resultQueue
        try:
            print("Training classifier "+key)
            labels = self.create_binary_labels(y, 1)
            classifier.fit(X, labels)
            cross_score = np.mean(cross_val_score(classifier, X, y, cv=10))
            print("Cross score is:")
            print(cross_score)
            print(classifier.classes_)
            print("Predicting and calculating probabilities...")
            conf_score = classifier.predict_proba(test_data)
            
            resultQueue.put((key, cross_score, conf_score))
        except Exception as e:
            print("OH NO IT'S AN EXCEPTION!")
            print(e)
        print("Done training classifier "+key)


    def create_class_specific_classifier(self, X, y, test_data, classifiers, filename):
        global resultQueue

        output_data = []
        conf_scores = OrderedDict()
        cross_scores = OrderedDict()
        cross_scores['name'] = filename
        resultQueue = Queue()

        #Build a list of classifiers in the order they were given to us. 
        #This will help reorganize everything after going through the ThreadPoolExecutor
        classNames = []
        for key in classifiers.keys():
            classNames.append(key)

        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for key in classifiers.keys():
                    #Pre-populate these scores so that the proper order is maintained
                    cross_scores[key] = None
                    conf_scores[key] = None
                    executor.submit(self.train, X, y, test_data, classifiers[key], key)
            

        while not resultQueue.empty():
            data = resultQueue.get();
            #We have some data!
            print("Collecting data for the results of "+str(data[0]))
            #(key, cross_score, conf_score)
            print("DATA 1 is:")
            print(data[1])
            cross_scores[data[0]] = data[1]
            conf_scores[data[0]] = data[2]

        print("Writing to output file...")
        for index, row in enumerate(test_data):
            output_row = []
            for name in classNames:
                output_row.append(conf_scores[name][index][1])
            output_row.append(self.get_final_label(output_row))
            output_data.append(output_row)

        # Write to output file
        with open(OUTPUT_DIR + '/' + filename + '.txt', 'w', newline='') as fp:
            a = csv.writer(fp, delimiter='\t')
            a.writerows(output_data)
        #The key is being prepended here. Not sure why...
        print("CROSS SCORE VALUES ARE:")
        print(dict(cross_scores))
        #print("CONF SCORE VALUES ARE:")
        #print(dict(conf_scores))
        return dict(cross_scores), dict(conf_scores)

    def get_final_label(self, conf_scores):
        #returns the index of the maximum conf score
        return conf_scores.index(max(conf_scores))+1;
        
    # Decision tree
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # Y : array-like, shape = [n_samples]

    def classifier_tree(self):
        return DecisionTreeClassifier()

    def classifier_bagging_trees(self, estimators):
        return BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)

    def classifier_random_forests(self, estimators):
        return RandomForestClassifier(n_estimators=estimators)

    def classifier_boosting(self, estimators):
        return AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=estimators)

    def classifier_logistic(self):
        return LogisticRegression()

    def classifier_randomization(self, estimators):
        estimators = estimators
        return ExtraTreesClassifier(estimators)
        
    def classifier_bayes_gaussian(self):
        return GaussianNB()

    def getClassifiers(self, classKeywords, estimators):
        classifiers = OrderedDict()
        i = 1
        for keyword in classKeywords:
            print("keyword1 is "+keyword)
            if keyword.startswith("gaussian"):
                classifiers[keyword] = self.classifier_bayes_gaussian()
            elif keyword.startswith("random"):
                classifiers[keyword] = self.classifier_randomization(estimators)
            elif keyword.startswith("logistic"):
                classifiers[keyword] = self.classifier_logistic()
            elif keyword.startswith("boosting"):
                classifiers[keyword] = self.classifier_boosting(estimators)
            elif keyword.startswith("forest"):
                classifiers[keyword] = self.classifier_random_forests(estimators)
            elif keyword.startswith("bagging"):
                classifiers[keyword] = self.classifier_bagging_trees(estimators)
            elif keyword.startswith("decision"):
                classifiers[keyword] = self.classifier_tree()
            i += 1
        print(classifiers)
        return classifiers

#scores = self.create_class_specific_classifier(X, y, test_data, gaussNB, "gaussNB")

    def writeScores(self, cross_scores, headers):
        headers.insert(0,"name")
        with open("scores.csv", 'a') as csvfile:
            #for i in range(0, len(cross_scores)):
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if(os.stat("scores.csv").st_size == 0):
                #Write top headers
                writer.writeheader()
            writer.writerow(cross_scores)


    def main(self):
        scores = []
        print("Loading files...")
        training_data = self.load_training_data()
        training_label = self.load_training_label()
        test_data = self.load_test_data()
        #test_data = [[float(y) for y in x] for x in test_data]
        print("Test data type:")
        print(type(test_data[0][0]))
        training_data = self.remove_missing_values(training_data)

        print("Doing imputation...")
        imp = Imputer(strategy='mean', axis=0)
        imp.fit(training_data, training_label)
        print("Taining data type:")
        print(type(training_data[0][0]))
        training_data = imp.transform(training_data)




        ####
        # There are 7 options for classifiers that you can use:
        # 1. gaussian
        # 2. random
        # 3. logistic
        # 4. forest
        # 5. bagging
        # 6. decision
        # 7. Logistic
        # 8. Gaussian
        #####

        filename = "244_100_rem"
        classStrings = ["logistic_1", "logistic_2", "logistic_3"]
        classifier_dict = self.getClassifiers(classStrings, 100)
        cross_scores, conf_scores = self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filename)
        print("Cross validation scores are...")
        print(cross_scores)
        self.writeScores(cross_scores, classStrings)



classification = classification()
classification.main()
