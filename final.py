import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def transform(Sequence):
	element = []
	n = len(Sequence)
	for amino_acid in codes:
		r = Sequence.count(amino_acid)
		composition = r
		element.append(composition)
	return element

train_data = pd.read_csv("train.csv")
X_train = train_data['Sequence'].apply(lambda x: transform(x))
y_train = train_data['Lable']


max_length = max(X_train.apply(lambda x: len(x)))
X_train = pad_sequences(X_train, max_length,padding='post',truncating='post')
# print(X_train.shape)

testing_data = pd.read_csv("test.csv")
X_test = testing_data['Sequence'].apply(lambda x: transform(x))
max_length = max(X_test.apply(lambda x: len(x)))
X_test = pad_sequences(X_test,max_length,padding='post',truncating='post')
# print(X_test.shape)

kf = KFold(10)

rf_class = RandomForestClassifier(n_estimators=20)
log_class = LogisticRegression()
svm_class = SVC(kernel='linear')
extra_tree_class = ExtraTreesClassifier(n_estimators = 5, criterion ='entropy', max_features = 2) 

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty='l1', dual=False))),
  ('classification', ExtraTreesClassifier())
])
clf.fit(X_train, y_train)

# print("\nExtra tree:")
accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
# print("Accuracy of extra is: " , accuracy)

Y_preds = clf.predict(X_test)
print(Y_preds)
results = pd.DataFrame({'ID':testing_data['ID'],'Label':Y_preds})
results.to_csv(r'final_result.csv',index=False,header=True)