import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def transform(Sequence):
	element = []
	n = len(Sequence)
	for amino_acid in codes:
		r = Sequence.count(amino_acid)
		composition = r/n
		element.append(composition)
	return element

train_data = pd.read_csv("train.csv")
X_train = train_data['Sequence'].apply(lambda x: transform(x))
print(X_train[0])
y_train = train_data['Lable']

max_length = max(X_train.transform(lambda x: len(x)))
X_train = pad_sequences(X_train, max_length,padding='post',truncating='post')
X_train = to_categorical(X_train)
n,nx,ny = X_train.shape
X_train = X_train.reshape(n,nx*ny)
# print(X)


testing_data = pd.read_csv("test.csv")
X_test = testing_data['Sequence'].apply(lambda x: transform(x))
max_length = max(X_test.apply(lambda x: len(x)))
X_test = pad_sequences(X_test,max_length,padding='post',truncating='post')
X_test = to_categorical(X_test)
n,nx,ny = X_test.shape
X_test = X_test.reshape(n,nx*ny)

# model = RandomForestClassifier(n_estimators=20,random_state=0)
# # model = LogisticRegression()
# model.fit(X_train, y_train)

# Y_preds = model.predict(X_test)
# print(Y_preds)
# results = pd.DataFrame({'ID':testing_data['ID'],'Label':Y_preds})
# results.to_csv(r'random_forest_amino_count.csv',index=False,header=True)
