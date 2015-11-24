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
from sknn.mlp import Classifier, Layer
from sklearn import svm

import csv
import numpy as np
import datetime
import concurrent.futures
from queue import Queue
import os
import sys
from itertools import combinations_with_replacement

from sknn.mlp import Classifier, Layer
from sklearn import svm

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
        except:
            print("OH NO IT'S AN EXCEPTION!")
            print(sys.exc_info())
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
            return self.classifier_boosting(estimators, ExtraTreesRegressor(estimators,kernel='linear'))
        if(bag):
            return self.classifier_bagging_trees(estimators, classifier=ExtraTreesRegressor(estimators))
            
        return ExtraTreesRegressor(estimators)

    def classifier_randomization(self, estimators, boost=False, bag=False):
        estimators = estimators
        if(boost):
            return self.classifier_boosting(estimators, ExtraTreesClassifier(estimators,kernel='linear'))
        if(bag):
            return self.classifier_bagging_trees(estimators, classifier=ExtraTreesClassifier(estimators))
            
        return ExtraTreesClassifier(estimators, criterion="entropy")
        
    def classifier_bayes_gaussian(self, boost=False, bag=False):
        if boost and bag:
            #return self.classificer_bagging_trees(10, self.classifier_boosting(GaussianNB()))
            return self.classifier_boosting(10, self.classifier_bagging(10, GaussianNB()))
        if(boost):
            return self.classifier_boosting(100, GaussianNB())
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


    def classifier_svm(self, boost=False, bag=False):
        print("SVM")
        if(boost):
            return self.classifier_boosting(10, svm.SVC(probability=True,kernel='linear'))
        if(bag):
            return self.classifier_bagging_trees(10, svm.SVC(probability=True,kernel='linear'))
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

    def getClassifiers(self, classKeywords, estimators=50, boost=False, bag=False):
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
            elif keyword.startswith("treereg"):
                classifiers[keyword] = self.classifier_tree_regressor(estimators,boost, bag)
            elif keyword.startswith("nn"):
                classifiers[keyword] = self.classifier_nn(estimators, boost, bag)
            elif keyword.startswith("svm"):
                classifiers[keyword] = self.classifier_svm(boost, bag)
            i += 1
        print(classifiers)
        return classifiers

#scores = self.create_class_specific_classifier(X, y, test_data, gaussNB, "gaussNB")

    def writeScores(self, cross_scores, auc_scores, headers):
        print("Writing cross_scores")
        print(cross_scores)
        cross_scores = dict(cross_scores)
        cross_scores["auc_1"] = auc_scores[0]
        cross_scores["auc_2"] = auc_scores[1]
        cross_scores["auc_3"] = auc_scores[2]

        headers.insert(0,"name")
        headers.append("auc_1")
        headers.append("auc_2")
        headers.append("auc_3")

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

        training_data = self.remove_missing_values(training_data)

        print("Doing imputation...")
        imp = Imputer(strategy='mean', axis=0)
        imp.fit(training_data, training_label)

        training_data = imp.transform(training_data)
        return training_data, training_label, test_data

    def makeROCPlot(self, filename, title, labels, roc_data):
        y = np.array(self.create_binary_label_matrix(labels))
        n_classes = y.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
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
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig("figs/"+filename+'.png',bbox_inches='tight')
        #plt.show()
        plt.clf()
        return roc_auc



    def main(self):
        global classStrings
        training_data, training_label, test_data = self.loadData()
        ####
        # There are 10 options for classifiers that you can use:
        # 1. gaussian
        # 2. random
        # 3. logistic
        # 4. forest
        # 5. bagging
        # 6. decision
        # 7. treereg
        # 8. nn
        # 9. svm
        # 10. boosting
        #####

        ### MODIFY THE LINES BELOW WITH THE APPROPRIATE VALUES# 
        title = "Gaussian Naive Bayes"
        filename = "111_gauss"
        classStrings = ["gaussian_1", "gaussian_2", "gaussian_3"]
        runParallel = True
        classifier_dict = self.getClassifiers(classStrings, 10, False, False)
        cross_scores, conf_scores, roc_data= self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filename, runParallel)

        auc_scores = self.makeROCPlot(title, filename, training_label, roc_data)

        self.writeScores(cross_scores,auc_scores, classStrings)


        #Uncomment the following to run a large section of tests at a time.
        #May require supervision - useful for finding bugs!
'''
        allClass = dict()
        allClass[1] = "gaussian"
        allClass[2] = "random"
        allClass[3] = "logistic"
        allClass[4] = "forest"
        allClass[6] = "decision"
        allClass[7] = "treereg"
        allClass[8] = "nn"
        allClass[9] = "svm"

        allClassList = [1,2,3,4,5,6,7,8,9]
        allClassCombos = list(combinations_with_replacement(allClassList, 3))
        start = allClassCombos.index((4,6,6))
        print(allClassCombos)
        for combo in allClassCombos:
            try:
                title = allClass[combo[0]]+", "+allClass[combo[1]]+", "+allClass[combo[2]]
                filename = str(combo[0])+str(combo[1])+str(combo[2])
                classStrings = [allClass[combo[0]]+"_1",allClass[combo[1]]+"_2",allClass[combo[2]]+"_3"]
                classifier_dict = self.getClassifiers(classStrings, 10, False, False)
                cross_scores, conf_scores, roc_data= self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filename)
                auc_scores = self.makeROCPlot(title, filename, training_label, roc_data)
                self.writeScores(cross_scores,auc_scores, classStrings)
            except Exception:

        #Run with boosting!
        start = allClassCombos.index((6,6,6))
        for combo in allClassCombos[start:]:
            try:
                title = str(allClass[combo[0]])+", "+str(allClass[combo[1]])+", "+str(allClass[combo[2]])+"_boosting"
                filename = str(combo[0])+str(combo[1])+str(combo[2])+"_boosting"
                classStrings = [allClass[combo[0]]+"_1",allClass[combo[1]]+"_2",allClass[combo[2]]+"_3"]
                classifier_dict = self.getClassifiers(classStrings, 50, True, False)
                cross_scores, conf_scores, roc_data= self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filename)
                auc_scores = self.makeROCPlot(title, filename, training_label, roc_data)
                self.writeScores(cross_scores,auc_scores, classStrings)
            except Exception:
                print("Exception occurred!")

        #Run with bagging!
        allClassCombos = list(combinations_with_replacement(allClassList, 3))
        allClassCombos = allClassCombos[5:]
        for combo in allClassCombos:
            try:
                title = str(allClass[combo[0]])+", "+str(allClass[combo[1]])+", "+str(allClass[combo[2]])+"_bagging"
                filename = str(combo[0])+str(combo[1])+str(combo[2])+"_bagging"
                classStrings = [allClass[combo[0]]+"_1",allClass[combo[1]]+"_2",allClass[combo[2]]+"_3"]
                classifier_dict = self.getClassifiers(classStrings, 50, False, True)
                cross_scores, conf_scores, roc_data= self.create_class_specific_classifier(training_data, training_label, test_data, classifier_dict, filename)
                auc_scores = self.makeROCPlot(title, filename, training_label, roc_data)
                self.writeScores(cross_scores,auc_scores, classStrings)
            except Exception:
                print("Exception occurred!")'''

classification = classification()
classification.main()
