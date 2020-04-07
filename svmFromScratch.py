import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os


###WORK IN PROGRESS###
style.use('ggplot')

data_dict = {-1:np.array([[1,7],[2,8],[3,8]]), 
            1:np.array([[5,1],[6,-1],[7,3]])}


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

        #support vectors yi(xi.w+b) = 1
        #until you hit a value close to 1, keep stepping
        #making more steps will give you a more accurate model, but will be more computational expensive
        #Can we run this in parallel? Unfortunately no - We need previous knowledge of each step
        step_sizes = [self.max_feature_value*.1, self.max_feature_value*.01, self.max_feature_value*.001]


        #b does not need to take as small of steps as w does
        #we do not need to take as many steps with b as we do w
        
        b_range_multiple = 5 
        b_multiple = 5

        #First element in vector W
        latest_optimum = self.max_feature_value*10

        for step in step_sizes: 
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                #more efficient than range
                #This can be ran in parallel for efficiency gains
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_multiple):
                    for transform in transforms:
                        w_t = w*transform
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(xi,w_t)+b) >= 1:
                                    found_option = False
                        
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    #w = [5,5]
                    #step = 1
                    #w - [step, step] = [4,4]
                    w = w - step
            #magnitudes
            norms = sorted([n for n in opt_dict])
            #opt dict is the smallest norm
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
        
        #Print Values of Data Dict
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi, ':', yi*(np.dot(self.w, xi)+self.b))

    def predict(self, features):
        """Gets sign of classifcation, plots it on ax"""
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])            
       
        return classification

    def visualize(self):
        """Visualizes SVM"""
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #hyperplane = x.w+b
        #v = hyperplane
        #positive sv = 1
        #negative sv = -1
        #decision = 0
        def hyperplane(x,w,b,v):
            """plots SVM hyperplane & Decision boundary hyperplane
               returns the point to plot on graph for human reference"""
            return (-w[0]*x-b+v) / w[1]

        #scales graph
        datarange = (self.min_feature_value*.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        #y axis coordinates
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max],[psv1, psv2], 'k')

        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max],[nsv1, nsv2], 'k')

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max],[db1, db2], 'y--')

        plt.show()

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10], [1,3], [3,4],[3,5],[5,5],[5,6],[6,-5],[5,8]]
for p in predict_us:
    svm.predict(p)

svm.visualize()