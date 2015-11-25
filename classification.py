from config import *

from collections import OrderedDict

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.linear_model.logistic import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

import csv
import numpy as np
import datetime
import concurrent.futures
from queue import Queue
import os


class classification:
    resultQueue = Queue()
    classStrings = ""

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

    #Creates labels of the form [1,0,0], [0,1,0]
    def create_binary_labels(self, classes, keyIndex):
        binary_labels = []
        for clas in classes:
            if int(clas) == int(keyIndex):
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        return binary_labels
    #Classes is list, like [1,2,3,2], output [100],[010],[001]...
    def create_binary_label_matrix(self, classes):
        binary_labels = []
        for row in classes:
            if int(row) == 1:
                binary_labels.append([1,0,0])
            if int(row) == 2:
                binary_labels.append([0,1,0])
            if int(row) == 3:
                binary_labels.append([0,0,1])

        return binary_labels


    def train(self, X, y, test_data, classifier, key,queue=True):
        global resultQueue
        try:
            print("Training classifier "+key)
            labels = self.create_binary_labels(y, classStrings.index(key)+1)
            classifier.fit(X, labels)
            cross_score = np.mean(cross_val_score(classifier, X, y, cv=10))
            print("Cross score is:")
            print(cross_score)
            print(classifier.classes_)
            print("Predicting and calculating probabilities...")
            conf_score = classifier.predict_proba(test_data)
            roc_data = classifier.predict_proba(X)
            print("ROC DATA IS:")
            print(roc_data)
            print("CONF DATA IS: ")
            print(conf_score)
            if queue:
                resultQueue.put((key, cross_score, conf_score, roc_data))
            else:
                return cross_score, conf_score, roc_data
        except Exception as e:
            print("OH NO IT'S AN EXCEPTION!")
            print(e)
        print("Done training classifier "+key)

    #Set parallel to false for easier debugging
    def create_class_specific_classifier(self, X, y, test_data, classifiers, filename, parallel=True):
        global resultQueue

        output_data = []
        conf_scores = OrderedDict()
        cross_scores = OrderedDict()
        cross_scores['name'] = filename
        roc_data = OrderedDict()
        resultQueue = Queue()

        #Build a list of classifiers in the order they were given to us. 
        #This will help reorganize everything after going through the ThreadPoolExecutor
        classNames = []
        for key in classifiers.keys():
            classNames.append(key)

        if(parallel):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                for key in classifiers.keys():
                        #Pre-populate these scores so that the proper order is maintained
                        cross_scores[key] = None
                        conf_scores[key] = None
                        roc_data[key] = None
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
                roc_data[data[0]] = data[3]
        else:
            for key in classifiers.keys():
                data = self.train(X, y, test_data, classifiers[key], key, False)
                cross_scores[key] = data[0]
                conf_scores[key] = data[1]
                roc_data[key] = data[2]

        print("Writing to output file...")
        for index, row in enumerate(test_data):
            output_row = []
            for key in classNames:
                output_row.append(conf_scores[key][index][1])
            output_row.append(self.get_final_label(output_row))
            output_data.append(output_row)

        #Create roc training data
        roc_data_arr = []
        for index, row in enumerate(X):
            roc_data_row = []
            for key in classNames:
                roc_data_row.append(roc_data[key][index][1])
            roc_data_arr.append(roc_data_row)

        # Write to output file
        with open(OUTPUT_DIR + '/' + filename + '.txt', 'w', newline='') as fp:
            a = csv.writer(fp, delimiter='\t')
            a.writerows(output_data)
        #The key is being prepended here. Not sure why...
        print("CROSS SCORE VALUES ARE:")
        print(dict(cross_scores))
        #print("CONF SCORE VALUES ARE:")
        #print(dict(conf_scores))
        #Returning output_data as conf_scores
        return cross_scores, conf_scores, np.array(roc_data_arr)

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
        if boost and bag:
            return self.classifier_boosting(estimators, self.classifier_bagging_trees(estimators, RandomForestClassifier(n_estimators=estimators)))
        if(boost):
            return self.classifier_boosting(estimators, RandomForestClassifier(n_estimators=estimators))
        if(bag):
            return self.classifier_bagging_trees(estimators, RandomForestClassifier(n_estimators=estimators))
        return RandomForestClassifier(n_estimators=estimators)


    def classifier_logistic(self, boost=False, bag=False):
        print("LOGISTIC")
        if(boost):
            #return self.classifier_boosting(10, LogisticRegression())
            #Workaround for boosting with LogisticRegression
            return self.classifier_boosting(10, SGDClassifier(loss='log'))
            
        if(bag):
            return self.classifier_bagging_trees(10, LogisticRegression())
        return LogisticRegression()

    def classifier_tree_regressor(self, estimators, boost=False, bag=False):
        estimators = estimators
        if boost and bag:
            return self.classifier_boosting(estimators, self.classifier_bagging(ExtraTreesRegressor(estimators)))
        if(boost):
            return self.classifier_boosting(estimators, ExtraTreesRegressor(estimators))
        if(bag):
            return self.classifier_bagging_trees(estimators, classifier=ExtraTreesRegressor(estimators))
            
        return ExtraTreesRegressor(estimators)

    def classifier_randomization(self, estimators, boost=False, bag=False):
        estimators = estimators
        if(boost):
            return self.classifier_boosting(estimators, ExtraTreesClassifier(estimators))
        if(bag):
            return self.classifier_bagging_trees(estimators, classifier=ExtraTreesClassifier(estimators))
            
        return ExtraTreesClassifier(estimators, criterion="entropy")
        
    def classifier_bayes_gaussian(self, boost=False, bag=False):
        if boost and bag:
            #return self.classificer_bagging_trees(10, self.classifier_boosting(GaussianNB()))
            return self.classifier_boosting(10, self.classifier_bagging(10, GaussianNB()))
        if(boost):
            return self.classifier_boosting(50, GaussianNB())
        if(bag):
            return self.classifier_bagging_trees(10, GaussianNB())
        return GaussianNB()


    def getClassifiers(self, classKeywords, estimators, boost=False, bag=False):
        classifiers = OrderedDict()
        i = 1
        for keyword in classKeywords:
            print("keyword1 is "+keyword)
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
            elif keyword.startswith("treereg"):
                classifiers[keyword] = self.classifier_tree_regressor(boost, bag)
            i += 1
        print(classifiers)
        return classifiers

#scores = self.create_class_specific_classifier(X, y, test_data, gaussNB, "gaussNB")

    def writeScores(self, cross_scores, headers):
        print(cross_scores)
        cross_scores = dict(cross_scores)
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

    def makeROCPlot(self, labels, roc_data):
        #y_score = conf_score
        #y = label_binarize(labels, classes=[1,2,3])
        print("Before binaryfying")
        print(labels)
        y = np.array(self.create_binary_label_matrix(labels))
        print("After binaryfying")
        print(y)
        n_classes = y.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        print("Y IS:")
        print(y)
        print("ROC DATA IS")
        print(roc_data)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], roc_data[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), roc_data.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],label='Average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i+1, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Gaussian ROC with Boosting, 50 estimators')
        plt.legend(loc="lower right")
        plt.show()

    def main(self):
        global classStrings
        training_data, training_label, test_data = self.loadData()
        ####
        # There are 7 options for classifiers that you can use:
        # 1. gaussian
        # 2. random
        # 3. logistic
        # 4. forest
        # 5. bagging
        # 6. decision
        # 7. treereg
        #####


        filename = "111_boost_50_rem"
        classStrings = ["gaussian_1", "gaussian_2", "gaussian_3"]
        classifier_dict = self.getClassifiers(classStrings, 50, True, False)
        cross_scores, conf_scores, roc_data= self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filename, True)
        #training_label = y (iris target)

        self.makeROCPlot(training_label, roc_data)
        print("Cross validation scores are...")
        print(cross_scores)
        self.writeScores(cross_scores, classStrings)


classification = classification()
classification.main()
