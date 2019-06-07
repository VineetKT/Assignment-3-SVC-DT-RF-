import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from pprint import pprint

'''
-----------------------------Description--------------------------------------------
Functions:
- dataimport() : reading the file and splitting it into features and label matrices
- train_split() : splitting the dataset into training and testing set
- Randomized_CV : Performs randomised cross-validation for a classifier
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
    Y = le.fit_transform(Y)         # Rocks are encoded as 1; Mines are encoded as 0
    return X, Y

def train_split(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1)
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test) 
    return X_train, X_test, Y_train, Y_test

def model_SVC(X, Y, kernel_):
    X_train, X_test, Y_train, Y_test = train_split(X, Y)
    model = svm.SVC(kernel= kernel_, C = 1, gamma = 1)
    model.fit(X_train,Y_train)
    predicted_SVC = model.predict(X_test)
    print('Prediction using SVC: \n', predicted_SVC)
    metrics_(Y_test, predicted_SVC)

def model_DT(X, Y, critr):
    X_train, X_test, Y_train, Y_test = train_split(X, Y)
    model = DecisionTreeClassifier(criterion = critr)
    model.fit(X_train, Y_train)
    predicted_DT = model.predict(X_test)
    print('Prediction using Decision Tree: ', predicted_DT)
    metrics_(Y_test, predicted_DT)

def Randomized_CV(clf):
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 150, num = 10)]  # Number of trees in random forest
    max_features = ['auto', 'sqrt']    # Number of features to consider at every split
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]  # Maximum number of levels in tree
    max_depth.append(None)
    min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
    min_samples_leaf = [1, 2, 4]   # Minimum number of samples required at each leaf node
    bootstrap = [True, False]  # Method of selecting samples for training each tree

    # Creating the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    # Random search of parameters 
    clf_random = RandomizedSearchCV(estimator= clf, param_distributions= random_grid,
                                n_iter = 100, scoring='neg_mean_absolute_error', 
                                cv = 2, verbose=0, random_state=42, n_jobs=-1, iid= None,
                                return_train_score=True)
    # Fit the random search model
    return clf_random

def model_RFClassifier(X, Y):
    X_train, X_test, Y_train, Y_test = train_split(X, Y)
    model = RandomForestClassifier()
    cv_model = Randomized_CV(model)
    cv_model.fit(X_train, Y_train)
    print('Best Parameters using Randomised CV: \n', cv_model.best_params_, '\n')
    predicted_RF = cv_model.predict(X_test)
    print('Prediction using Random Forest Classification: ', predicted_RF)
    metrics_(Y_test, predicted_RF)

def metrics_(test, predicted):
    print('Confusion Metrix:\n', metrics.confusion_matrix(test, predicted))
    print('Classification Report:\n', metrics.classification_report(test, predicted))
    
def main():
    X, Y = dataimport('sonar.all-data')
    X_train, X_test, Y_train, Y_test = train_split(X, Y)
    #print(DecisionTreeClassifier.get_params(DecisionTreeClassifier).keys())
    
    print('---------Classification using SVC Algorithm:---------- \n')
    model_SVC(X, Y, 'linear')
    print('\n')
    
    #print(DecisionTreeClassifier.get_params().keys())
    
    print('---------Classification using Decision Tree Algorithm:---------- \n')
    print('________________Entropy method_________________')
    model_DT(X,Y, 'entropy')

    print('________________Gini Index method____________________')
    model_DT(X,Y, 'gini')
    print('\n')
    
    
    print('---------Classification using Random Forest Classification:------------ \n')
    model_RFClassifier(X,Y)
    

if __name__ == "__main__":
    main()