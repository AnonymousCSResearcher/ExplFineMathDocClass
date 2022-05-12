import csv

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
import numpy as np

import json
import pickle

# SET PARAMETERS
nr_digits = 2 # 2 or 5 (e.g., 76 vs. 76M12)
labs_key = 'mscs'

#sources = ['titles','texts','refs','titles_text','titles_refs','texts_refs','titles_texts_refs']#
sources = ['keyword','title','text','keyword title','keyword text','title text','keyword title text']
src_idxs = {}
src_idxs[labs_key] = 0
src_idxs[sources[0]] = 1
src_idxs[sources[1]] = 2
src_idxs[sources[2]] = 3
src_idxs[sources[3]] = 4
src_idxs[sources[4]] = 5
src_idxs[sources[5]] = 6
src_idxs[sources[6]] = 7
source_selected = sources[4]

encodings = ['tfidf']
encoding = encodings[0]

eval_modes = ['split','cv']
eval_mode = eval_modes[0] # split
test_size = 0.3 # if eval_mode = 'split'
cv = 3 # 3/10 # if eval_mode = 'cv'

# FUNCTION
def parse_file(csv_reader):
    mscs = []
    titles = []
    texts = []
    refs = []
    titles_texts = []
    titles_refs = []
    texts_refs = []
    titles_texts_refs = []
    for row in csv_reader:
        # extract msc
        msc = row[1][:nr_digits]
        mscs.append(msc)
        # extract title
        title = row[2]
        titles.append(title)
        # extract text
        text = row[3]
        texts.append(text)
        # extract refs
        ref = row[4]
        refs.append(ref)
        # unite title and text
        title_text = title + " " + text
        titles_texts.append(title_text)
        # unite title and refs
        title_ref = title + " " + ref
        titles_refs.append(title_ref)
        # unite text and refs
        text_ref = text + " " + ref
        texts_refs.append(text_ref)
        # unite title, text, and refs
        title_text_ref = title + " " + text + " " + ref
        titles_texts_refs.append(title_text_ref)
    return mscs,titles,texts,refs,titles_texts,titles_refs,texts_refs,titles_texts_refs

# OPEN FILES
print("OPEN FILES")

#root_path = "F:\\zbMath/"
root_path = "C:\\Users/phili/Downloads/"
eval_path = 'evaluation/classification/'

file_path = "out"

data_dict = {}
with open(root_path + file_path + ".csv", 'r', encoding='utf8') as f:
    csv_reader = csv.reader(f, delimiter=",")
    # PARSE FILES
    print("PARSE FILES")
    # parse
    print("parse")
    data_dict[file_path] = parse_file(csv_reader)

# with open(root_path + "data_dict.pkl",'wb') as f:
#     pickle.dump(data_dict,f)

# with open(root_path + "data_dict.pkl",'rb') as f:
#     data_dict = pickle.load(f)

# VECTORIZE
print("VECTORIZE")

# encode
print("encode")

vect_dict = {}
data_idx = 0  # mscs
# msc binary for y vector
vect_dict['mscs_bin'] = LabelBinarizer().fit_transform(data_dict[file_path][src_idxs[labs_key]])
# msc label strings for classification report
vect_dict['mscs_str'] = data_dict[file_path][src_idxs[labs_key]]

# select specific sources for evaluation
#sources = [sources[0],sources[1],sources[2]]

for source in [source_selected]:# keyword title
    print(source)

    vect_dict[source] = {}

    if encoding == encodings[0]:
        vect_dict[source][encoding] = TfidfVectorizer().fit_transform(data_dict[file_path][src_idxs[source]])
    # if encoding == encodings[1]:
    #     vect_dict[source][encoding] = MultiLabelBinarizer().fit_transform(data_dict[file_path][src_idxs[source]])

del data_dict

# with open(root_path + "vect_dict.pkl",'wb') as f:
#     pickle.dump(vect_dict,f)

# with open(root_path + "vect_dict.pkl",'rb') as f:
#     vect_dict = pickle.load(f)

# unite features
#print("unite features")
#texts_and_refs = FeatureUnion([("pca", PCA(n_components=1)),
#...                       ("svd", TruncatedSVD(n_components=2))])\
#    .fit_transform([vect_dict['texts'][encoding],vect_dict['ref_mscs'][encoding]])

# CLASSIFY
print("CLASSIFY")

# msc label strings for classification report
labs = vect_dict['mscs_str']

eval_dict = {}
for source in [source_selected]:# keyword title
    print(source)
    eval_dict[source] = {}

    X, y = vect_dict[source][encoding], vect_dict['mscs_str']

    # train-test split
    if eval_mode == 'split':

        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        # fit classifier
        print("fit classifier")
        #classifier = OneVsRestClassifier(LogisticRegression(verbose=1)).fit(X_train,y_train)
        classifier = LogisticRegression(verbose=1).fit(X_train,y_train)

        # evaluate classifier
        print("evaluate classifier")
        y_pred = classifier.predict(X_test)
        y_true = y_test
        eval_dict[source][encoding] = {}
        eval_dict[source][encoding]['accuracy'] = metrics.accuracy_score(y_true=y_true,y_pred=y_pred)
        eval_dict[source][encoding]['precision'] = metrics.precision_score(y_true=y_true,y_pred=y_pred,average='weighted')
        eval_dict[source][encoding]['recall'] = metrics.recall_score(y_true=y_true,y_pred=y_pred,average='weighted')

        probas_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        try:
            eval_dict[source][encoding]['precision_recall_curve'] = metrics.precision_recall_curve(y_true=y_true,probas_pred=probas_pred)
        except Exception as e:
            eval_dict[source][encoding]['precision_recall_curve'] = e

        eval_dict[source][encoding]['f1'] = metrics.f1_score(y_true=y_true,y_pred=y_pred,average='weighted')
        #except:
        #    print("no f1-score for " + source + " and " + encoding)

        eval_dict[source][encoding]['classification_report'] = metrics.classification_report(y_true=y_true, y_pred=y_pred,target_names=labs)

    # cross-validation
    if eval_mode == 'cv':
        # fit classifier
        print("fit classifier")
        #classifier = OneVsRestClassifier(LogisticRegression(verbose=1)).fit(X,y)
        classifier = LogisticRegression(verbose=0).fit(X,y)
        # evaluate classifier
        print("evaluate classifier")
        #accuracy = np.mean(cross_val_score(classifier, X, y, cv=cv))
        scores = cross_validate(classifier, X, y, cv=cv, scoring='f1_micro')
        #eval_dict[source][encoding] = accuracy
        eval_dict[source][encoding] = scores
        #print(source + "," + encoding + "," + str(accuracy))
        print(source + "," + encoding + "," + str(scores))

#########
# RESULTS
#########

with open(eval_path + "eval_dict_" + source_selected + ".json",'wb') as f:
    json.dump(eval_dict,f)

print("end")