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
from sknn.mlp import Classifier, Layer
from sklearn import svm

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
        

    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # Y : array-like, shape = [n_samples]

    def classifier_tree(self, boost=False, bag=False):
        return DecisionTreeClassifier()

    def classifier_bagging_trees(self, estimators, classifier=DecisionTreeClassifier()):
        print("BAGGING!")
        return BaggingClassifier(classifier, estimators, 0.67, 1.0, True, True)

    def classifier_boosting(self, estimators, classifier=DecisionTreeClassifier()):
        print("BOOSTING!")
        return AdaBoostClassifier(classifier, n_estimators=estimators)

    def classifier_random_forests(self, estimators, boost=False, bag=False):
        print("RANDOM FOREST")
        if(boost):
            return self.classifier_boosting(estimators, RandomForestClassifier(n_estimators=estimators))
        if(bag):
            return self.classifier_bagging_trees(estimators, RandomForestClassifier(n_estimators=estimators))
        return RandomForestClassifier(n_estimators=estimators)


    def classifier_logistic(self, boost=False, bag=False):
        print("LOGISTIC")
        if(boost):
            return self.classifier_boosting(10, LogisticRegression())
        if(bag):
            return self.classifier_bagging_trees(10, LogisticRegression())
        return LogisticRegression()

    def classifier_randomization(self, estimators, boost=False, bag=False):
        estimators = estimators
        print("RANDOM!")
        if(boost):
            return self.classifier_boosting(estimators, ExtraTreesClassifier(estimators))
        if(bag):
            return self.classifier_bagging_trees(estimators, classifier=ExtraTreesClassifier(estimators))
            
        return ExtraTreesClassifier(estimators)
        
    def classifier_bayes_gaussian(self, boost=False, bag=False):
        print("GAUSSIAN")
        if(boost):
            return self.classifier_boosting(10, GaussianNB())
        if(bag):
            return self.classifier_bagging_trees(10, GaussianNB())
        return GaussianNB()

    def classifier_svm(self, boost=False, bag=False):
        print("SVM")
        if(boost):
            return self.classifier_boosting(10, svm.SVC(probability=True))
        if(bag):
            return self.classifier_bagging_trees(10, svm.SVC(probability=True))
        return svm.SVC(probability=True)

    def classifier_nn(self, estimators, boost=False, bag=False):
        # Not working yet.
        print("Neural Network")
        nn = Classifier(
            layers=[
                Layer("Maxout", units=estimators, pieces=2),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=25)
        if(boost):
            return self.classifier_boosting(10, nn)
        if(bag):
            return self.classifier_bagging_trees(10, nn)
        return nn


    def getClassifiers(self, classKeywords, estimators, boost=False, bag=False):
        classifiers = OrderedDict()
        i = 1
        for keyword in classKeywords:
            print("keyword" + str(i) + " is " + keyword)
            if keyword.startswith("gaussian"):
                classifiers[keyword] = self.classifier_bayes_gaussian(boost, bag)
            elif keyword.startswith("random"):
                classifiers[keyword] = self.classifier_randomization(estimators, boost, bag)
            elif keyword.startswith("logistic"):
                classifiers[keyword] = self.classifier_logistic(boost, bag)
            elif keyword.startswith("boosting"):
                classifiers[keyword] = self.classifier_boosting(estimators, boost, bag)
            elif keyword.startswith("forest"):
                classifiers[keyword] = self.classifier_random_forests(estimators, boost, bag)
            elif keyword.startswith("bagging"):
                classifiers[keyword] = self.classifier_bagging_trees(estimators, boost, bag)
            elif keyword.startswith("decision"):
                classifiers[keyword] = self.classifier_tree(boost, bag)
            elif keyword.startswith("nn"):
                classifiers[keyword] = self.classifier_nn(estimators, boost, bag)
            elif keyword.startswith("svm"):
                classifiers[keyword] = self.classifier_svm(boost, bag)
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

    #Loads test and training data from files
    #returns data and data labels
    def loadData(self):
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
        print("Training data type:")
        print(type(training_data[0][0]))
        training_data = imp.transform(training_data)
        return training_data, training_label, test_data

    def main(self):

        training_data, training_label, test_data = self.loadData()
        ####
        # There are 7 options for classifiers that you can use:
        # 1. gaussian
        # 2. random
        # 3. logistic
        # 4. forest
        # 5. bagging
        # 6. decision
        # 7. nn
        # 8. svm
        # 9. boosting
        #####

        # filenames = ["111_boost_rem", "123_rain", "keep_guessing", "neuros"]
        # classStrings = [["gaussian_1", "gaussian_2", "gaussian_3"],
        #                 ["forest_1", "forest_2", "forest_3"],
        #                 ["logistic_1", "logistic_2", "logistic_3"],
        #                 ["nn_1", "nn_2", "nn_3"]]
        # for index, classString in enumerate(classStrings):
        #     classifier_dict = self.getClassifiers(classString, 400, False, False)
        #     cross_scores, conf_scores = self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filenames[index])
        #     print("Cross validation scores are...")
        #     print(cross_scores)
        #     self.writeScores(cross_scores, classString)

        # filenames = ["111_random_rain", "123_random_gaussianrain", "random_guess", "neuros"]
        # classStrings = [["random_1", "forest_2", "forest_3"],
        #                 ["random_1", "gaussian_2", "forest_3"],
        #                 ["random_1", "gaussian_2", "gaussian_3"]]
        # for index, classString in enumerate(classStrings):
        #     classifier_dict = self.getClassifiers(classString, 400, False, False)
        #     cross_scores, conf_scores = self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filenames[index])
        #     print("Cross validation scores are...")
        #     print(cross_scores)
        #     self.writeScores(cross_scores, classString)


        filenames = ["111_support"]
        classStrings = [["svm_1", "svm_2", "svm_3"]]
        for index, classString in enumerate(classStrings):
            classifier_dict = self.getClassifiers(classString, 400, False, False)
            cross_scores, conf_scores = self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filenames[index])
            print("Cross validation scores are...")
            print(cross_scores)
            self.writeScores(cross_scores, classString)

        # pip install scikit-neuralnetwork



classification = classification()
classification.main()
