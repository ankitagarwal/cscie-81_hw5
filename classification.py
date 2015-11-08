from config import *

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import Imputer

import csv

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
                    X[row_ind][col_ind] = 2
        return X

    def create_binary_label(self, classes, label):
        binary_labels = []
        for clas in classes:
            if int(clas) == int(label):
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        return binary_labels

    def create_class_specific_classifier(self, X, y, classifier1, classifier2, classifier3):
        labels = self.create_binary_label(y, 1)
        classifier1.fit(X, labels)

        labels = self.create_binary_label(y, 2)
        classifier2.fit(X, labels)

        labels = self.create_binary_label(y, 3)
        classifier3.fit(X, labels)

        print(classifier1.predict_proba(X[:1]))


    # Do plain vanilla CART
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # Y : array-like, shape = [n_samples]
    def classifier_tree(self, X, y):
        tree1 = DecisionTreeClassifier()
        tree2 = DecisionTreeClassifier()
        tree3 = DecisionTreeClassifier()
        self.create_class_specific_classifier(X, y, tree1, tree2, tree3)

    def main(self):
        print("Loading files...")
        training_data = self.load_training_data()
        training_label = self.load_training_label()
        training_data = self.remove_missing_values(training_data)

        # print("Doing imputation...")
        # imp = Imputer(strategy='most_frequent', axis=0)
        # imp.fit(training_data, training_label)

        print("Running tree classifiers...")
        self.classifier_tree(training_data, training_label)

classification = classification()
classification.main()