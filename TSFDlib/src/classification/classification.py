from sklearn import preprocessing, tree
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import pydotplus
import tools
reload(tools)
from tools import *


FEATURES = '../data/extracted_features/FEATURES.tab'
LABELS = '../data/extracted_features/LABELS.tab'
LABELS_DESCRIPTION = '../data/extracted_features/LABELS_DESCRIPTION.tab'
ACTIVITIES = ['Walk', 'Run', 'Stand', 'Sit']

# Read files
features = np.array(read_csv(FEATURES, delimiter=';', header=None))
labels = np.array(read_csv(LABELS, delimiter=';', header=None)).ravel()
labels_description = np.array(read_csv(LABELS_DESCRIPTION, delimiter=';', header=None))

# # Data normalization
# features = preprocessing.normalize(features, norm='l2')

# Find best classifier and get results
find_best_classifier(features, labels, ACTIVITIES, labels_description)
