import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

amino_acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
codes = {}
for i in range(len(amino_acids)):
	codes[amino_acids[i]] = i+1;

def transform(seq):
	l=[]
	for p in seq:
		if p in amino_acids:
			l.append(codes[p])
		else:
			l.append(0)
	return l

train_data = pd.read_csv("train.csv")
X = train_data['Sequence'].apply(lambda x: transform(x))
Y = train_data['Lable']

max_length = max(X.transform(lambda x: len(x)))
X = pad_sequences(X, max_length,padding='post',truncating='post')
X = to_categorical(X)
n,nx,ny = X.shape
X = X.reshape(n,nx*ny)
# print(X)


testing_data = pd.read_csv("test.csv")
X_test = testing_data['Sequence'].apply(lambda x: transform(x))
max_length = max(X_test.apply(lambda x: len(x)))
X_test = pad_sequences(X_test,max_length,padding='post',truncating='post')
X_test = to_categorical(X_test)
n,nx,ny = X_test.shape
X_test = X_test.reshape(n,nx*ny)

model = RandomForestClassifier(n_estimators=20,random_state=0)
# model = svm.SVC()
model.fit(X,Y)

Y_preds = model.predict(X_test)
print(Y_preds)
results = pd.DataFrame({'ID':testing_data['ID'],'Label':Y_preds})
results.to_csv(r'results.csv',index=False,header=True)
