
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

diabetes_dataset = pd.read_csv('/content/diabetes.csv')

diabetes_dataset.head()

diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_train_pre = classifier.predict(X_train)
training_data_acc = accuracy_score(X_train_pre, Y_train)

print(training_data_acc)

X_test_pre = classifier.predict(X_test)
test_data_acc = accuracy_score(X_test_pre, Y_test)

print(test_data_acc)

input_data = (1,189,60,23,846,30.1,0.398,59)

input_data_asnumpy_array = np.asarray(input_data)

input_data_reshape = input_data_asnumpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

prediction = classifier.predict(std_data)

print(prediction)

if (prediction[0] == 0):
  print('Not diabetic')
else:
  print('Diabetic')

