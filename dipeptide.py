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

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# d = {'A' : 0, 'C' : 1, 'D': 2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
codes2 = []
d = {}

def transform(Sequence):
	print("Here here here")
	element = []
	n = len(Sequence)
	for amino_acid in codes2:
		count = 0
		for i in range(n - 1):
			if(amino_acid.find(Sequence[i]) != -1 && amino_acid.find(Sequence[i + 1]) != -1):
				count += 1
		element.append(count)
	print(len(element))
	return element


for i in range(20):
	for j in range(i, 20):
		codes2.append(codes[i] + codes[j])

train_data = pd.read_csv("train.csv")
X_train = train_data['Sequence'].apply(lambda x: transform(x))
y_train = train_data['Lable']


max_length = max(X_train.apply(lambda x: len(x)))
X_train = pad_sequences(X_train, max_length,padding='post',truncating='post')
print(X_train.shape)
# X_train = to_categorical(X_train)
# n,nx,ny = X_train.shape
# X_train = X_train.reshape(n,nx*ny)
# print(X)


testing_data = pd.read_csv("test.csv")
X_test = testing_data['Sequence'].apply(lambda x: transform(x))
max_length = max(X_test.apply(lambda x: len(x)))
X_test = pad_sequences(X_test,max_length,padding='post',truncating='post')
print(X_test.shape)
# X_test = to_categorical(X_test)
# n,nx,ny = X_test.shape
# X_test = X_test.reshape(n,nx*ny)

# model = ExtraTreesClassifier(n_estimators = 5, criterion ='entropy', max_features = 2) 
# # model = LogisticRegression()
# model.fit(X_train, y_train)

# Y_preds = model.predict(X_test)
# print(Y_preds)
# results = pd.DataFrame({'ID':testing_data['ID'],'Label':Y_preds})
# results.to_csv(r'extra_tree.csv',index=False,header=True)


kf = KFold(10)

rf_class = RandomForestClassifier(n_estimators=20)
log_class = LogisticRegression()
svm_class = SVC(kernel='linear')
extra_tree_class = ExtraTreesClassifier(n_estimators = 5, criterion ='entropy', max_features = 2) 

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
# clf = ExtraTreesClassifier(n_estimators=50)
# clf = clf.fit(X_train, y_train)
# model = SelectFromModel(clf, prefit=True)
# X_new = model.transform(X_train)
# X_new.shape 

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty='l1', dual=False))),
  ('classification', ExtraTreesClassifier())
])
clf.fit(X_train, y_train)

# print("Random Forests: ")
# accuracy = cross_val_score(svm_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
# print("Accuracy of Random Forests is: " , accuracy)
 
# print("\n\nSVM:")
# accuracy = cross_val_score(svm_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
# print("Accuracy of SVM is: " , accuracy)
 
# print("\n\nLog:")
# accuracy = cross_val_score(log_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
# print("Accuracy of SVM is: " , accuracy)

print("\nExtra tree:")
accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of extra is: " , accuracy)

Y_preds = clf.predict(X_test)
print(Y_preds)
results = pd.DataFrame({'ID':testing_data['ID'],'Label':Y_preds})
results.to_csv(r'current.csv',index=False,header=True)