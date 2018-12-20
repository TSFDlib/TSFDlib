import numpy as np
import matplotlib.pyplot as plt
import pandletstools as pt
import novainstrumentation as ni
import seaborn
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing, tree
import pydotplus

def plotfft(s, fmax, doplot=False):
    """ This functions computes the fft of a signal, returning the frequency
    and their magnitude values.

    Parameters
    ----------
    s: array-like
      the input signal.
    fmax: int
      the sampling frequency.
    doplot: boolean
      a variable to indicate whether the plot is done or not.

    Returns
    -------
    f: array-like
      the frequency values (xx axis)
    fs: array-like
      the amplitude of the frequency values (yy axis)
    """

    fs = np.abs(np.fft.fft(s))
    f = np.linspace(0, fmax / 2, len(s) / 2)
    # if doplot:
    #     pl.plot(f[1:len(s) / 2], fs[1:len(s) / 2])
    return (f[1:len(s) / 2].copy(), fs[1:len(s) / 2].copy())

def plot_confusion_matrix(cm, activities, title='Confusion matrix', cmap=plt.cm.Blues):

    target_names = activities
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    for i, cas in enumerate(cm):
        for j, c in enumerate(cas):
            if c > 0:
                plt.text(j-.2, i+.2, c, fontsize=20)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, fontsize=20)
    plt.yticks(tick_marks, target_names, fontsize=20)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.grid('off')


def get_confusion_matrix(cm, activities):

    # Compute confusion matrix
    np.set_printoptions(precision=2)
    fig = plt.figure('Confusion Matrix')
    plot_confusion_matrix(cm, activities, title='Confusion Matrix\n')
    if not os.path.exists('../data/results'):
        os.makedirs('../data/results')
    fig.savefig('../data/results/Confusion Matrix.png', bbox_inches='tight')

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm_normalized.round(decimals=3)
    fig = plt.figure('Normalized Confusion Matrix')
    fig.set_size_inches(10, 8)
    plot_confusion_matrix(cm_normalized, activities, title='Normalized Confusion Matrix')
    fig.savefig('../data/results/Normalized Confusion Matrix.png', bbox_inches='tight')
    plt.show()


def get_DT_pseudocode(tree, feature_names):
    """
    This functions gives the decision tree pseudo-code
    :param tree: (sklearn DecisionTreeClassifier)
           decision tree
    :param feature_names: (string array)
           the names of all existing features
    :return: txt file with the decision tree pseudo-code
    """
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    if not os.path.exists('../data/results'):
        os.makedirs('../data/results')
    decision_tree = open('../data/results/Decision_Tree.txt', 'w')

    def recurse(left, right, threshold, features, node):
        if threshold[node] != -2:
            decision_tree.write("if ( " + str(features[node]) + " <= " + str(threshold[node]) + " ) {" + '\n')
            if left[node] != -1:
                    recurse(left, right, threshold, features, left[node])
            if right[node] != -1:
                    recurse(left, right, threshold, features, right[node])
            decision_tree.write("" + '\n')
        else:
            if np.argmax(value[node]) == 0:
                c = 'WALK'
            if np.argmax(value[node]) == 1:
                c = 'RUN'
            if np.argmax(value[node]) == 2:
                c = 'STILL'
            decision_tree.write("return PhysicalActivity." + str(c) + '; \n} \n')

    recurse(left, right, threshold, features, 0)
    decision_tree.close()


def find_best_classifier(features, labels, classes, labels_description):
    """
    This function performs the classification of the given features using several classifiers. From the obtained results 
    the classifier which best fits the data and gives the best result is chosen and the respective confusion matrix is 
    showed.
    :param  features: (array)
            features
    :param  labels: (array)
            features respective labels
    :param  classes: (str list)
            names of the existing classes
    """
    # Classifiers
    names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "SVM", "AdaBoost", "Naive Bayes", "QDA"]
    classifiers = [
        KNeighborsClassifier(5),
        DecisionTreeClassifier(max_depth=5, min_samples_split=len(features)/10),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2),
        svm.SVC(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    best = 0
    best_classifier = None
    best_y_test_pred = None
    best_clf = None

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    for n, c in zip(names, classifiers):
        print n
        scores = cross_val_score(c, features, labels, cv=10)
        print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

        # Train the classifier
        c.fit(X_train, y_train.ravel())

        # Predict test data
        y_test_predict = c.predict(X_test)

        # Get the classification accuracy
        accuracy = accuracy_score(y_test, y_test_predict)
        print("Accuracy: " + str(accuracy) + '%')
        print('-----------------------------------------')
        if np.mean([scores.mean(), accuracy]) > best:
            best_classifier = n
            best_y_test_pred = y_test_predict
            best = np.mean([scores.mean(), accuracy])
            best_clf = c

    print('******** Best Classifier: ' + str(best_classifier) + ' ********')
    # Get confusion matrix
    plt.close('all')
    cm = confusion_matrix(y_test, best_y_test_pred)
    plt.figure()
    plot_confusion_matrix(cm, classes, title='Confusion matrix')
    plt.figure()
    plot_confusion_matrix(cm, classes, title='Normalized Confusion matrix')
    plt.show()

    if best_classifier == "Decision Tree":
        # Export decision rules
        dot_data = tree.export_graphviz(best_clf, out_file=None, feature_names=labels_description, class_names=classes,
                                        filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("../data/results/Decision_Tree.pdf")