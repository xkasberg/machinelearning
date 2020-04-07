import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os

style.use('ggplot')

data_dict = {-1:np.array([[1,7],[2,8],[3,8]]), 
            1:np.array([[5,1],[6,-1],[7,3]])}

print(data_dict)

class Support_Vector_Machine:
    """Vladimir Vapnik, Support Vector Machine Algorithm
       Represented in OOP"""

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        """The larger the data set, the longer the training process"""
        self.data = data
        #{||W||: [w, b]}
        opt_dict = {}
        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]
        all_data = []
        for yi in self.data: 
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None


        step_sizes = [self.max_feature_value*.1, self.max_feature_value*.01, self.max_feature_value*.001]

        #b does not need to take as small of steps as w does
        b_range_multiple = 5 
        #we do not need to take as many steps with b as we do w

        b_multiple = 5

        #First element in vector W
        latest_optimum = self.max_feature_value*10

        for step in step_sizes: 
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                #more efficient than range
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_multiple):
                    for transform in transforms:
                        w_t = w*transformation
                        #innocent until proven guilty 
                        found_option = True
                        #weakest link in the SVM alogorithm
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i





        pass

    def predict(self, data):
        classification = np.sign(np.dot(np.array(features), self.w)+b)
        return classification

