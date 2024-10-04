import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset and model
diabetes_dataset = pd.read_csv('/Users/prajwal/praju/Project/DiabetesPrediction/diabetes.csv')

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

def predict_diabetes(request):
    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        
        # Convert input to numpy array
        input_data = [float(i) for i in input_data.split(',')]
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
        
        # Standardize input data
        std_data = scaler.transform(input_data_reshape)
        
        # Make prediction
        prediction = classifier.predict(std_data)[0]
        
        if prediction == 0:
            result = 'Not Diabetic'
        else:
            result = 'Diabetic'
        
        context = {'result': result}
        return render(request, 'result.html', context)
    
    return render(request, 'predict.html')
