import numpy as np 
from sklearn import neighbors, svm
from sklearn.model_selection import cross_validate, train_test_split
import pandas as pd
import os

accuracies = []

for i in range(25):
    df = pd.read_csv(os.getcwd()+'/datasets/breast-cancer-wisconsin.data')
    #Replacing it with -999999 algorithms treat the data as an Outlier rather than dropping it, depends on the algorithm
    #Depends on how much data would be dropped if you dropped null values
    df.replace('?', -99999, inplace=True)

    #EDA 
    #Useless Data?
    df.drop(['id'], 1, inplace=True)
    #print(df.head())

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    #print(accuracy)
    accuracies.append(accuracy)

print("average accuracy: " +str((sum(accuracies)/len(accuracies))))
# #Should be a unique measure not existing in dataset
# example_measure = np.array([4,2,1,1,1,2,3,2,1])
# print("Before: ")
# print(example_measure)
# print("Length: "+str(len(example_measure)))
# print(example_measure.shape)
# example_measure = example_measure.reshape(1,-1)
# #converts example_measure into a list of lists, the shame format as our dataset

# #First argument
# #Refers to the number of samples, in this case the number of lists in a list of lists,  
# #Second argument 
# #When reshaping an array, the new shape must contain the same number of elements as the old shape
# #Ie, the products of the two shapes dimensions must be equal
# #When using a -1, the dimension corresponding to the -1 will be 
# #the product of the dimensions of the original array 
# #divided by the product of the dimensions given 
# #to reshape so as to maintain the same number of elements
# print("After: ")
# print(example_measure)
# prediction = clf.predict(example_measure)
# print(prediction)



