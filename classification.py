from config import *

from collections import OrderedDict

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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
                data.append(row[:-1])
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
            print("CONF SCORE IS: ")
            print(conf_score)
            resultQueue.put((key, cross_score, conf_score))
        except Exception as e:
            print("OH NO IT'S AN EXCEPTION!")
            print(e)
        print("Done training classifier "+key)


    def create_class_specific_classifier(self, X, y, test_data, scores, classifiers, filename):
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
        return dict(cross_scores)

    def get_final_label(self, conf_scores):

        # TODO Try other methods?
        #returns the index of the maximum conf score
        return conf_scores.index(max(conf_scores))+1;
        
    # Decision tree
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # Y : array-like, shape = [n_samples]
    def classifier_tree(self, X, y, test_data, scores):
        print("Running tree classifiers...")
        classifier_dict = OrderedDict()
        classifier_dict['tree1'] = DecisionTreeClassifier()
        classifier_dict['tree2'] = DecisionTreeClassifier()
        classifier_dict['tree3'] = DecisionTreeClassifier()
        scores = self.create_class_specific_classifier(X, y, test_data, scores, classifier_dict, "Decision_tree")
        return ['name', 'tree1', 'tree2', 'tree3'], scores


    def classifier_bagging_trees(self, X, y, test_data, scores):
        estimators = 10
        for i in range(2, 20):
            print("Running bagging tree classifiers with " + str(estimators) + " estimators...")
            classifier_dict = OrderedDict()
            classifier_dict['bagging1'] = BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)
            classifier_dict['bagging2'] = BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)
            classifier_dict['bagging3'] = BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)
            scores = self.create_class_specific_classifier(X, y, test_data, scores, classifier_dict, "Bagging_tree_" + str(estimators))
            estimators *= i
        return ['name', 'bagging1', 'bagging2', 'bagging3'], scores

    def classifier_bagging_trees_and_decision(self, X, y, test_data, scores):
        # Based on ACU scores we noticed decision tree is doing quite well for first and second classifier and bagging for third.
        estimators = 10
        classifier_dict = OrderedDict()
        print("Running mix of bagging tree and decision tree classifiers with " + str(estimators) + " estimators...")
        classifier_dict['tree1'] = DecisionTreeClassifier()
        classifier_dict['tree2'] = DecisionTreeClassifier()
        classifier_dict['bagging3'] = BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)
        scores = self.create_class_specific_classifier(X, y, test_data, scores, classifier_dict, "Bagging_tree_decision_")
        return ['name', 'tree1', 'tree2', 'bagging3'], scores

    def classifier_random_forests(self, X, y, test_data, scores):
        estimators = 10
        for i in range(2, 6):
            forest_dict = OrderedDict()
            print("Running Random forest classifiers with " + str(estimators) + " estimators...")
            forest_dict['forest1'] = RandomForestClassifier(n_estimators=estimators)
            forest_dict['forest2'] = RandomForestClassifier(n_estimators=estimators)
            forest_dict['forest3'] = RandomForestClassifier(n_estimators=estimators)
            scores = self.create_class_specific_classifier(X, y, test_data, scores, forest_dict, "forest_" + str(estimators))
            estimators += (i * 10)
        return ['name', 'forest1', 'forest2', 'forest3'], scores

    def classifier_boosting(self, X, y, test_data, scores):
        estimators = 10
        for i in range(2, 6):
            boosting_dict = OrderedDict()
            print("Running Boosting classifiers with " + str(estimators) + " estimators...")
            boosting_dict['boosting1'] = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=estimators)
            boosting_dict['boosting2'] = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=estimators)
            boosting_dict['boosting3'] = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=estimators)
            scores = self.create_class_specific_classifier(X, y, test_data, scores, boosting_dict, "boosting_" + str(estimators))
            estimators += (i * 10)
        return ['name', 'boosting1', 'boosting2', 'boosting3'], scores

    def classifier_logistic(self, X, y, test_data, scores):
        # This is not working yet.
        logistic_regression = OrderedDict()
        logistic_regression['logistic1'] = LogisticRegression()
        logistic_regression['logistic1'] = LogisticRegression()
        logistic_regression['logistic1'] = LogisticRegression()
        scores = self.create_class_specific_classifier(X, y, test_data, scores, logistic_regression, "regression_")
        return ['name', 'logistic1', 'logistic2', 'logistic3'], scores

    # Do Randomization
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # Y : array-like, shape = [n_samples]
    def classifier_randomization(self, X, y, test_data, scores):
        estimators = 10
        for i in range(2, 8):
            print("Running random classifiers with " + str(estimators) + " estimators...")
            random_dict = OrderedDict()
            random_dict['random1'] = ExtraTreesClassifier(200)
            random_dict['random2'] = ExtraTreesClassifier(200)
            random_dict['random3'] = ExtraTreesClassifier(200)
            scores = self.create_class_specific_classifier(X, y, test_data, scores, random_dict, "random_" + str(estimators))
            estimators += (i * 10)
        return ['name', 'random1', 'random2', 'random3'], scores


    def classifier_random_forests(self, X, y, test_data, scores):
        estimators = 10
        for i in range(2, 6):
            forest_dict = OrderedDict()
            print("Running Random forest classifiers with " + str(estimators) + " estimators...")
            forest_dict['forest1'] = RandomForestClassifier(n_estimators=estimators)
            forest_dict['forest2'] = RandomForestClassifier(n_estimators=estimators)
            forest_dict['forest3'] = RandomForestClassifier(n_estimators=estimators)
            scores = self.create_class_specific_classifier(X, y, test_data, scores, forest_dict, "forest_" + str(estimators))
            estimators += (i * 10)
        return ['name', 'forest1', 'forest2', 'forest3'], scores
        
    #def classifier_bayes_gaussian(self, X, y, test_data, scores):


    def main(self):
        scores = []
        print("Loading files...")
        training_data = self.load_training_data()
        training_label = self.load_training_label()
        test_data = self.load_test_data()
        training_data = self.remove_missing_values(training_data)

        print("Doing imputation...")
        imp = Imputer(strategy='most_frequent', axis=0)
        imp.fit(training_data, training_label)
        training_data = imp.transform(training_data)

        scoreHeaders = []
        scores = []
        treeData = self.classifier_tree(training_data, training_label, test_data, scores)
        scoreHeaders.append(treeData[0])
        scores.append(treeData[1])

        bagTree = self.classifier_bagging_trees(training_data, training_label, test_data, scores)
        scoreHeaders.append(bagTree[0])
        scores.append(bagTree[1])

        bagTreeDecision = self.classifier_bagging_trees_and_decision(training_data, training_label, test_data, scores)
        scoreHeaders.append(bagTreeDecision[0])
        scores.append(bagTreeDecision[1])

        randomForest = self.classifier_random_forests(training_data, training_label, test_data, scores)
        scoreHeaders.append(randomForest[0])
        scoreHeaders.append(randomForest[1])
        # scores = self.classifier_tree(training_data, training_label, test_data, scores)
        # scores = self.classifier_bagging_trees(training_data, training_label, test_data, scores)
        # scores = self.classifier_bagging_trees_and_decision(training_data, training_label, test_data, scores)
        # scores = self.classifier_random_forests(training_data, training_label, test_data, scores)
        # scores = self.classifier_boosting(training_data, training_label, test_data, scores)
        # scores = self.classifier_logistic(training_data, training_label, test_data, scores)
        randomization = self.classifier_randomization(training_data, training_label, test_data, scores)
        scoreHeaders.append(randomization[0])
        scores.append(randomization[1])

        print("Cross validation scores are...")
        print(scores)

        with open("scores.csv", 'a') as csvfile:
            for i in range(0, len(scores)):
                writer = csv.DictWriter(csvfile, fieldnames=scoreHeaders[i])
                writer.writeheader()
                writer.writerow(scores[i])

classification = classification()
classification.main()
