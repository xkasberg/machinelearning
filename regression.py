import pandas as pd  
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


pd.options.mode.chained_assignment = None 


#Get Data
response_df = quandl.get('WIKI/GOOGL')
print("Response_df: ")
print(response_df.head())
print("Last Response Date: ")
print(response_df.index[-1])
response_df = response_df[['Adj. Open', 'Adj. Close', 'Adj. High','Adj. Low','Adj. Volume']]

#Create some Features
response_df['HL_PCT'] = (response_df['Adj. High'] - response_df['Adj. Close']) / response_df['Adj. Close'] * 100.00
response_df['PCT_Change'] = (response_df['Adj. Close'] - response_df['Adj. Open']) / response_df['Adj. Open']* 100.00

features_df = response_df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

#Clean Data
print("features_df nan values before cleaning: " +str(features_df.isna().sum()))
features_df.fillna(-99999, inplace=True)
print("feautres_df nan values after cleaning: " +str(features_df.isna().sum()))

#Calculate forecast set length
forecast_out = int(math.ceil(0.1*len(features_df)))
print("Forecast_out: " +str(forecast_out))
forecast_col = 'Adj. Close'
features_df['label'] = features_df[forecast_col].shift(-forecast_out)
print('')
print("Show Features[Label] Shifted ")
print(features_df['label'][-(forecast_out+10):])
print('')
print("Features_df Head:")
print('')
print(features_df.head())
print('')

#Features
#Convert DF into Array
X = np.array(features_df.drop(['label'],1))
print(X)
print('')
#Normalize Data
X = preprocessing.scale(X)
print("Processed X: ")
print(X)

print("X[:-forecast_out]")
print(X)
X_lately = X[-forecast_out:]
print('')
print("X_Lately: ")
print(X_lately)
print('')
X = X[:-forecast_out]


#labels
features_df.dropna(inplace=True)
y = np.array(features_df['label'])

print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


#Select Classifer / Algorithm for prediction

# classifier = LinearRegression()
# classifier.fit(X_train, y_train)
## Save Classifier to pickle
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(classifier, f)

pickle_in = open('linearregression.pickle', 'rb')
classifier = pickle.load(pickle_in)

#accuracy squared_error - directionally accurate, similar to confidence in 2 dimensional data
accuracy = classifier.score(X_test, y_test)
print("Accuracy: " + str(accuracy))

#takes value or an array
forecast_set = classifier.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

features_df['Forecast'] = np.nan





#Chart
last_date = features_df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day 

for i in forecast_set: 
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    features_df.loc[next_date] = [np.nan for _ in range(len(features_df.columns)-1)] + [i]

features_df['Adj. Close'].plot()
features_df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
