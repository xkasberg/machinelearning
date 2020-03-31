import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
from collections import Counter
from math import sqrt
import warnings
import pandas as pd
import random
import os

style.use('fivethirtyeight')


#plot1 = [1,3]
#plot2 = [2, 5]
# euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)

#dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}

#new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1], s=100, color=i)

# plt.show()

def euclidean_distance(xs, ys):
    """Xs and Ys are lists"""
    elements = []
    for x, y in zip(xs,ys):
        ele = (x - y)**2
        elements.append(ele)
    elements = sum(elements)
    #print(sqrt(elements))
    return sqrt(elements)

def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        #Compares predict feature with every feature in dataset
        for feature in data[group]:
            e_distance = euclidean_distance(feature, predict)
            #e_distance = np.linalg.norm(np.array(feature)-np.array(predict))
            distances.append([e_distance, group])
    
    #print("Distances: "+str(distances))
    votes = [i[1] for i in sorted(distances)[:k]]
    #print("Votes: " +str(votes))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    #print("Vote Result: "+str(vote_result)+', Confidence: '+str(confidence))
    return vote_result, confidence

def accuracy(correct, total):
    return correct / total


#result = k_nearest_neighbours(dataset, new_features)
#print(result)


#euclidean_distance(plot1, plot2)


# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1], s=100, color=result)

# plt.show()


#print(euclidean_distance)
accuracies = []
for i in range(25):
    df = pd.read_csv(os.getcwd()+'/datasets/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    #print(df.head())
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)
    #print(full_data[:10])

    test_size = .20
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    #first 20% of data
    train_data = full_data[:-int(test_size*len(full_data))]
    #last 20% of data
    test_data = full_data[-int(test_size*len(full_data)):]

    correct = 0
    total = 0

    for i in train_data:
        #gets the label, the last value, either a 2, or a 4
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        #gets the label, the last value, either a 2, or a 4
        test_set[i[-1]].append(i[:-1])

    for group in test_set: 
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct += 1

            total += 1

    #print("Accuracy: " +str(round(accuracy(correct, total),4)*100.00)+'%')
    accuracies.append(accuracy(correct, total))

print("average accuracy: " +str(sum(accuracies) / len(accuracies)))