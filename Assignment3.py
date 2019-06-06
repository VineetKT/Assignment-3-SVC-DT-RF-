import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

'''
-----------------------------Description--------------------------------------------
Functions:
- dataimport() : reading the file and splitting it into features and label matrices
- train_split() : splitting the dataset into training and testing set
- model_SVC() : Classification model using SVC method
- model_DT() : Classification model using Decision Tree classification
- model_RFClassifier() : Classification model using Random Forest lassification
- metrics() : Getting various accuracy measures

variables:
- filename : name of the input file
- le : Lebel Encoder variable
- sc : standard feature scaling variable
- critr : criteria to decide the DT classification
- predicted_SVC : Predicted data on Test set using SVC
- predicted_DT :   "  using Decision Tree
- predicted_RF :   "  using Random Forest
-------------------------------------------------------------------------------------
'''

def dataimport(filename):
    data = pd.read_csv(filename, sep = ',', header = None)
    data_X = data.drop([60], axis = 1)
    X = np.array(data_X)
    Y = data[60]
    
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Y)
    return X, Y

def train_split(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test) 
    return X_train, X_test, Y_train, Y_test

def model_SVC(X, Y):
    X_train, X_test, Y_train, Y_test = train_split(X, Y)
    model = svm.SVC(kernel='linear', C = 1, gamma = 1)
    model.fit(X_train,Y_train)
    predicted_SVC = model.predict(X_test)
    print('Prediction using SVC: \n', predicted_SVC)
    metrics_(Y_test, predicted_SVC)

def model_DT(X, Y, critr):
    X_train, X_test, Y_train, Y_test = train_split(X, Y)
    model = DecisionTreeClassifier(criterion = critr, max_depth = 3, min_samples_leaf = 5)
    model.fit(X_train, Y_train)
    predicted_DT = model.predict(X_test)
    print('Prediction using Decision Tree: ', predicted_DT)
    metrics_(Y_test, predicted_DT)

def model_RFClassifier(X, Y):
    X_train, X_test, Y_train, Y_test = train_split(X, Y)
    model = RandomForestClassifier(n_estimators=20, bootstrap = True, max_depth= 20, max_features = 'sqrt')  
    model.fit(X_train, Y_train)  
    predicted_RF = model.predict(X_test)
    print('Prediction using Random Forest Classification: ', predicted_RF)
    metrics_(Y_test, predicted_RF)

def metrics_(test, predicted):
    print('Confusion Metrix: ', metrics.confusion_matrix(test, predicted))
    print('Accuracy : ', metrics.accuracy_score(test, predicted)*100, ' %')
    print('Precision : ', metrics.precision_score(test, predicted)*100, ' %')
    print('Recall : ', metrics.recall_score(test, predicted)*100, ' %')
    print('Mean Absolute Error:', metrics.mean_absolute_error(test, predicted)*100, '%')  
    print('Mean Squared Error:', metrics.mean_squared_error(test, predicted)*100, '%')  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test, predicted))*100, '%')

def main():
    X, Y = dataimport('sonar.all-data')    # data reading in features and label columns
    X_train, X_test, Y_train, Y_test = train_split(X, Y)

    ## ---------------- SVC --------------------##
    print('---------Classification using ecision Tree Algorithm:---------- \n')
    model_SVC(X, Y)
    print('\n')

    ##--------------Decision Tree--------------##
    print('---------Classification using ecision Tree Algorithm:---------- \n')
    print('Entropy method--------------------------------')
    model_DT(X,Y, 'entropy')

    print('Gini Index method-----------------------------')
    model_DT(X,Y, 'gini')
    print('\n')

    ##--------------Random Forest--------------##
    print('---------Classification using Random Forest Classification:------------ \n')
    model_RFClassifier(X,Y)

if __name__ == "__main__":
    main()