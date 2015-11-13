from config import *

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import Imputer

import csv
import numpy as np
import datetime
from multiprocessing import Pool

class classification:

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

    def create_class_specific_classifier(self, X, y, test_data, scores, classifier1, classifier2, classifier3, filename):
        output_data = []

        print("Training first classifier...")
        labels = self.create_binary_labels(y, 1)
        classifier1.fit(X, labels)
        cross_score1 = np.mean(cross_val_score(classifier1, X, y, cv=10))
        print(classifier1.classes_)

        print("Training second classifier...")
        labels = self.create_binary_labels(y, 2)
        classifier2.fit(X, labels)
        cross_score2 = np.mean(cross_val_score(classifier2, X, y, cv=10))
        print(classifier2.classes_)

        print("Training third classifier...")
        labels = self.create_binary_labels(y, 3)
        classifier3.fit(X, labels)
        cross_score3 = np.mean(cross_val_score(classifier3, X, y, cv=10))
        print(classifier3.classes_)

        print("Predicting and calculating probabilities...")
        conf_scores1 = classifier1.predict_proba(test_data)
        conf_scores2 = classifier2.predict_proba(test_data)
        conf_scores3 = classifier3.predict_proba(test_data)

        print("Writing to output file...")
        for index, row in enumerate(test_data):
            output_row = []
            output_row.append(conf_scores1[index][1])
            output_row.append(conf_scores2[index][1])
            output_row.append(conf_scores3[index][1])
            output_row.append(self.get_final_label(conf_scores1[index][1], conf_scores2[index][1], conf_scores3[index][1]))
            output_data.append(output_row)

        # Write to output file
        with open(OUTPUT_DIR + '/' + filename + '.txt', 'w', newline='') as fp:
            a = csv.writer(fp, delimiter='\t')
            a.writerows(output_data)

        scores.append({"name": filename, "classifier 1": cross_score1, "classifier 2": cross_score2, "classifier 3": cross_score3})
        return scores

    def get_final_label(self, score1, score2, score3):
        # TODO Try other methods?
        max_val = max(float(score1), float(score2), float(score3))
        if (max_val == float(score1)):
            return 1
        elif (max_val == float(score2)):
            return 2
        elif(max_val == float(score3)):
            return 3
        else:
            raise Exception('something is seriously wrong, this should never happen.')

    # Decision tree
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # Y : array-like, shape = [n_samples]
    def classifier_tree(self, X, y, test_data, scores):
        print("Running tree classifiers...")
        tree1 = DecisionTreeClassifier()
        tree2 = DecisionTreeClassifier()
        tree3 = DecisionTreeClassifier()
        return self.create_class_specific_classifier(X, y, test_data, scores, tree1, tree2, tree3, "Decision_tree")

    def classifier_bagging_trees(self, X, y, test_data, scores):
        estimators = 10
        for i in range(2, 20):
            print("Running bagging tree classifiers with " + str(estimators) + " estimators...")
            bagging1 = BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)
            bagging2 = BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)
            bagging3 = BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)
            scores = self.create_class_specific_classifier(X, y, test_data, scores, bagging1, bagging2, bagging3, "Bagging_tree_" + str(estimators))
            estimators *= i
        return scores

    def classifier_bagging_trees_and_decision(self, X, y, test_data, scores):
        # Based on ACU scores we noticed decision tree is doing quite well for first and second classifier and bagging for third.
        estimators = 60
        print("Running mix of bagging tree and decision tree classifiers with " + str(estimators) + " estimators...")
        tree1 = DecisionTreeClassifier()
        tree2 = DecisionTreeClassifier()
        bagging3 = BaggingClassifier(DecisionTreeClassifier(), estimators, 0.67, 1.0, True, True)
        scores = self.create_class_specific_classifier(X, y, test_data, scores, tree1, tree2, bagging3, "Bagging_tree_decision_" + str(estimators))
        return scores

    def classifier_random_forests(self, X, y, test_data, scores):
        estimators = 10
        for i in range(2, 6):
            print("Running Random forest classifiers with " + str(estimators) + " estimators...")
            forest1 = RandomForestClassifier(n_estimators=estimators)
            forest2 = RandomForestClassifier(n_estimators=estimators)
            forest3 = RandomForestClassifier(n_estimators=estimators)
            scores = self.create_class_specific_classifier(X, y, test_data, scores, forest1, forest2, forest3, "forest_" + str(estimators))
            estimators += (i * 10)
        return scores

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

        # scores = self.classifier_tree(training_data, training_label, test_data, scores)
        # scores = self.classifier_bagging_trees(training_data, training_label, test_data, scores)
        # scores = self.classifier_bagging_trees_and_decision(training_data, training_label, test_data, scores)
        scores = self.classifier_random_forests(training_data, training_label, test_data, scores)

        print("Cross validation scores are...")
        print(scores)

        with open("scores.csv", 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["name", "classifier 1", "classifier 2", "classifier 3"])
            writer.writeheader()
            for data in scores:
                writer.writerow(data)

classification = classification()
classification.main()