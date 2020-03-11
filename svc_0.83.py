import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")

y_train = np.array(training_data["Lable"])
print(y_train.shape)

X_train = training_data["Sequence"]
# X_train = pd.get_dummies(X_train)
X_train = np.array(X_train)
# print("Training ", X_train)

X_test = testing_data["Sequence"]
# X_test = pd.get_dummies(X_test)
X_test = np.array(X_test)
# print("Testing", X_test)

codes = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

X_train_encoded = []
for element in X_train:
	element_copy = []
	element = list(element)
	for char in element:
		char = codes[char]
		element_copy.append(char)
	X_train_encoded.append(element_copy)
# print(X_train_encoded)
X_train = X_train_encoded

X_test_encoded = []
for element in X_test:
	element_copy = []
	element = list(element)
	for char in element:
		char = codes[char]
		element_copy.append(char)
	X_test_encoded.append(element_copy)
# print(X_test_encoded)
X_test = X_test_encoded

from keras.preprocessing.sequence import pad_sequences

max_length = 100
train_pad = pad_sequences(X_train_encoded, maxlen=max_length, padding='post', truncating='post')
test_pad = pad_sequences(X_test_encoded, maxlen=max_length, padding='post', truncating='post')

print(train_pad.shape, test_pad.shape)

from keras.utils import to_categorical

train_ohe = to_categorical(train_pad)
test_ohe = to_categorical(test_pad)

print(train_ohe.shape, test_ohe.shape) 


model = SVC(kernel='linear', C=0.1)
# model = LinearRegression()

nsamples, nx, ny = train_ohe.shape
train_ohe = train_ohe.reshape((nsamples,nx*ny))

model.fit(train_ohe, y_train)

nsamples, nx, ny = test_ohe.shape
test_ohe = test_ohe.reshape((nsamples,nx*ny))
y_pred = model.predict(test_ohe)
print(y_pred)

import csv
count = 3551
with open('result_linear_svc_c0.1.csv', 'w', newline='\n') as file:
	writer = csv.writer(file)
	writer.writerow(["ID", "Label"])
	for element in y_pred:
		writer.writerow([count, element])
		count += 1
		