import pandas as pd
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, multilabel_confusion_matrix, auc

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Prepare our dataset of 1794 images of digits
#No cleaning is required as it comes from Sci-Kit
digits = load_digits()
images, labels = digits.images, digits.target
n_classes = 10
images = images.reshape(-1, 8*8)
labels = digits.target

print("Number of  Classes: " +str(n_classes))
#Split our data into training and testing arrays
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = .20, random_state = 4) 

#Decision Trees
#Creates an instance of the Decision Tree Classifier
dt = DecisionTreeClassifier()
print('Decision Tree Classifier')
#Fits our instance on the Training Data
dt = dt.fit(x_train, y_train)
#Our predictions array
predictions = dt.predict(x_test)
print("First 10 Predictions: " +str(predictions[:10]))
print("First 10 Labels: " +str(y_test[:10]))
#Now lets dive into some metrics:
#Accuracy is the total correct labels / total predictions 
#It is an incomplete measure of the competence of the model as 
#Accuracy fails to consider the distribution/variance of the dataset
dt_accuracy = round(dt.score(x_test, y_test), 4)* 100
print("Decision Tree Sci-Kit Accuracy: " +str(dt_accuracy)+"%")


dt_precision, dt_recall, dt_fscore, dt_support = precision_recall_fscore_support(y_true = y_test, y_pred=predictions)
#Precision improves on accuracy as it is the number of true positives, 
#or correctly classified data divided by 
# the correct classifications + the incorrect classifications'
#ie: tp / (tp + fp)')
#where tp is the  True  Positive Rate,  FP is the False Positive Rate

print("Decision Tree Sci-Kit Precision Digit 0: " +str(dt_precision[0]))
# print("Decision Tree Precision Digit 1: " +str(dt_precision[1]))
# print("Decision Tree Precision Digit 2: " +str(dt_precision[2]))
# print("Decision Tree Precision Digit 3: " +str(dt_precision[3]))
# print("Decision Tree Precision Digit 4: " +str(dt_precision[4]))
# print("Decision Tree Precision Digit 5: " +str(dt_precision[5]))
# print("Decision Tree Precision Digit 6: " +str(dt_precision[6]))
# print("Decision Tree Precision Digit 7: " +str(dt_precision[7]))
# print("Decision Tree Precision Digit 8: " +str(dt_precision[8]))
# print("Decision Tree Precision Digit 9: " +str(dt_precision[9]))
# print('')

#Recall is the ability of the model to find all the positive samples
#tp / tp + fn
#where fn is False Negative
print("Decision Tree Sci-Kit Recall Digit 0: " +str(dt_recall[0]))
# print("Decision Tree Recall Digit 1: " +str(dt_recall[1]))
# print("Decision Tree Recall Digit 2: " +str(dt_recall[2]))
# print("Decision Tree Recall Digit 3: " +str(dt_recall[3]))
# print("Decision Tree Recall Digit 4: " +str(dt_recall[4]))
# print("Decision Tree Recall Digit 5: " +str(dt_recall[5]))
# print("Decision Tree Recall Digit 6: " +str(dt_recall[6]))
# print("Decision Tree Recall Digit 7: " +str(dt_recall[7]))
# print("Decision Tree Recall Digit 8: " +str(dt_recall[8]))
# print("Decision Tree Recall Digit 9: " +str(dt_recall[9]))
# print('')
#F-Beta Score can be interpreted as a weighted harmonic mean of the precision and recall, where F reaches its best value at 1, worst score at 0")
#The Harmonic Mean is the Reciprocal of the Arithmetic Mean of the Reciprocals")
#example: consider an array of: [1, 4, 4] 
#([1/1 + 1/4 + 1/4]) / 3 = 1.5 / 3
# 3/1.5 = 2

print("Decision Tree Sci-Kit F-Beta Score Digit 0: " +str(dt_fscore[0]))
# print("Decision Tree F-Beta Score Digit 1: " +str(dt_fscore[1]))
# print("Decision Tree F-Beta Score Digit 2: " +str(dt_fscore[2]))
# print("Decision Tree F-Beta Score Digit 3: " +str(dt_fscore[3]))
# print("Decision Tree F-Beta Score Digit 4: " +str(dt_fscore[4]))
# print("Decision Tree F-Beta Score Digit 5: " +str(dt_fscore[5]))
# print("Decision Tree F-Beta Score Digit 6: " +str(dt_fscore[6]))
# print("Decision Tree F-Beta Score Digit 7: " +str(dt_fscore[7]))
# print("Decision Tree F-Beta Score Digit 8: " +str(dt_fscore[8]))
# print("Decision Tree F-Beta Score Digit 9: " +str(dt_fscore[9]))
#print('')

#The ROC curve tests the tp / fp rate over different decision thresholds, 
#ie: It is useful to see how well your classifier can separate 
#positive and negative examples, and identify the best 'Threshold' for seperating them 
#Values above the Baseline, ie 50% and closer to 1 are "better"
#Values below the Baseline and closer to zero are worse

#Decision Trees provide no rank, and therefoe no 'Threshold' and thus cannot be tuned 
#The ROC is a point rather than a line in the ROC Vector Space For Decision Trees

#We can get our TP & FP Rate for each class 
#by calculating the confusion Matrix
confusion_matrix  = multilabel_confusion_matrix(y_test, predictions)

def metrics(confusion_matrix, class_n):
    """
    confusion_matrix is a multi label confusion matrix, class_n is the number of different labels
    returns the occurence of FP, FN, TP, TN
    """
    FP = confusion_matrix[class_n][0][1]
    FN = confusion_matrix[class_n][1][0]
    TP = confusion_matrix[class_n][1][1]
    TN = confusion_matrix[class_n][0][0]

    return FP, FN, TP, TN


metrics_dict = {}
for i in range(10):
    FP, FN, TP, TN = metrics(confusion_matrix, i)
    metrics_dict[i] = FP, FN, TP, TN

fpr = {}
tpr = {}
for i in range(n_classes):
    fpr[i], tpr[i] = (metrics_dict[i][0] / (metrics_dict[i][0] + metrics_dict[i][3])), (metrics_dict[i][2] / (metrics_dict[i][2] + metrics_dict[i][1]))


print("Decision Tree Precision Digit 0: " + str((metrics_dict[0][2] / (metrics_dict[0][0] + metrics_dict[0][2])))+"%")

#Plots the ROC Points for each Feature with the Decision Tree Algorithm
# plt.figure()
# colours = plt.cm.rainbow(np.linspace(0, 1, 10))
# for i, c in zip(range(10), colours):
#     plt.scatter(fpr[i], tpr[i], color=colours[i], label=f'ROC Point Digit:  {i}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


print('')
# Random Forest is a common 'Base' Model in an ensemble approach as  
#Random Forest samples over features rather than data points 
#to arrive at less correlated models & more robust to missing data
#n_estimators is the number of trees in the forest 
#max_features is the number of features to consider when splitting
rf = RandomForestClassifier(n_estimators=20)
rf = rf.fit(x_train, y_train)
print("Random Forest Classifier")
print("Model Classes: " +str(rf.n_classes_))
print("Dataset Features: " +str(rf.n_features_))
print("Default Features to decide a split voter: "+str(np.sqrt(rf.n_features_)))

rf_predictions = rf.predict(x_test)
print("First 10 Predictions: " +str(rf_predictions[:10]))
print("First 10 Labels: " +str(y_test[:10]))

print("Random Forest Accuracy: " +str(round(rf.score(x_test, y_test), 2))+"%")
rf_precision, rf_recall, rf_fscore, rf_support = precision_recall_fscore_support(y_true = y_test, y_pred=rf_predictions)
print("Random Forest Sci-Kit Precision Digit 0: " +str(rf_precision[0]))
print("Random Forest Sci-Kit Recall Digit 0: " +str(rf_recall[0]))
print("Random Forest Sci-Kit F Beta Score Digit 0: " +str(rf_fscore[0]))
print("Random Forest Sci-Kit Support Digit 0: " +str(rf_support[0]))


#We can plot a Random Forest's ROC Curve as it is an Algorithm that provides ranks
#Baseline 'Worst case' Probabilities
r_probs = [0 for _ in range(len(y_test))]
rf_probs = rf.predict_proba(x_test)
rf_probs = rf_probs[:, 1]


r_auc = roc_auc_score(y_test, r_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

#Plots the ROC Points for each Feature with the Decision Tree Algorithm
# plt.figure()
# colours = plt.cm.rainbow(np.linspace(0, 1, 10))
# for i, c in zip(range(10), colours):
#     plt.scatter(fpr[i], tpr[i], color=colours[i], label=f'ROC Point Digit:  {i}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

#We can improve on with generic decision tree's with random forests
#Now we will improve on the decision tree with different Ensemble approaches 



# #Bagging - Decision Tree
# bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=.5, max_features=1.0, n_estimators = 20)
# print(bg.fit(x_train, y_train))
# print("Bagging Accuracy: " + str(round(bg.score(x_test, y_test), 2)*100)+"%")


# #Boosting - Adaptive Boosting - Decision Tree
# #learning rate controls how much each model attributes to the ensemble on each approach, muust be adjusted for overfit models
# adaBoost = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators = 10, learning_rate=1)
# print(adaBoost.fit(x_train, y_train))
# print("Bagging Accuracy, 10 estimators, 100% learning rate: " + str(round(adaBoost.score(x_test, y_test), 2)*100)+"%")

# #Boosting - Adaptive Boosting - Decision Tree
# #learning rate controls how much each model attributes to the ensemble on each approach, muust be adjusted for overfit models
# adaBoost = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators = 5, learning_rate=1)
# print(adaBoost.fit(x_train, y_train))
# print("Bagging Accuracy, 5 estimators: " + str(round(adaBoost.score(x_test, y_test), 2)*100)+"%")


# #Bagging - Random Forest
# bg = BaggingClassifier(RandomForestClassifier(), max_samples=.5, max_features=1.0, n_estimators = 20)
# print(bg.fit(x_train, y_train))
# print("Bagging Accuracy: " + str(round(bg.score(x_test, y_test), 2))+"%")


# #Boosting - Adaptive Boosting - Random Forest
# #learning rate controls how much each model attributes to the ensemble on each approach, muust be adjusted for overfit models
# adaBoost = AdaBoostClassifier(RandomForestClassifier(), n_estimators = 10, learning_rate=1)
# print(adaBoost.fit(x_train, y_train))
# print("Bagging Accuracy, 10 estimators, 100% learning rate: " + str(round(adaBoost.score(x_test, y_test), 2)*100)+"%")

# #Boosting - Adaptive Boosting - Random Forest
# #learning rate controls how much each model attributes to the ensemble on each approach, muust be adjusted for overfit models
# adaBoost = AdaBoostClassifier(RandomForestClassifier(), n_estimators = 5, learning_rate=1)
# print(adaBoost.fit(x_train, y_train))
# print("Bagging Accuracy, 5 estimators, 100% learning rate: " + str(round(adaBoost.score(x_test, y_test), 2)*100)+"%")


# #Stacking - Heterogenerous Model Ensemble 
# lr = LogisticRegression()
# dt = DecisionTreeClassifier()
# svm = SVC(kernel = 'poly', degree = 2)

# evc = VotingClassifier(estimators=[('lr', lr), ('dt', dt), ('svm', svm)], voting= 'hard')
# print(evc.fit(x_train, y_train))
# print("Voting Classifier Accuracy: " + str(round(evc.score(x_test, y_test), 2)*100)+"%")




"""FORMULAS"""
#Sensitivity = hit rate = recall = TPRate
# TPR = TP/(TP+FN)

# #Specifity, TNRate
# TNR = TN/(TN+FP)

# #Precision
# PPV = TP/(TP+FP)

# #Negative Predicted Value
# NPV = TN/(TN+FN)

# #Fall out or False Positive Rate
# FPR = FP / (FP + TN)

# #False Negative Rate
# FNR = FN / (TP + FN)

# #False Discovery Rate
# FDR = FP / (TP + FP)

# #Acurracy 
# Accuracy = (TP + TN) / (TP + FP + FN + TN)