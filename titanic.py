# Own approach towards the Titanic problem in Kaggle competition

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# machine learning algorithm library
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# hyperparameter tuning technique-Grid Search
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("C:/Users/13745/Desktop/WT2/Work/train23.csv")

# First we drop the column cabin since most of the data are missing
data.drop(['Cabin'], axis=1, inplace=True)
# Second we drop the column Ticket since the data is messy, which is 
# hard to do feature engineering
data.drop(['Ticket'], axis=1, inplace=True)

# Next we show how many family members each passenger have
data['Family_member'] = data['SibSp'] + data['Parch'] + 1
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Next we divide the fare into different groups -- low, medium, high
def cat_fare(price):
    if price<10:
        return 'Low'
    elif price>=10 and price<=50:
        return 'Medium'
    return 'High'

data['Price_range'] = data['Fare'].apply(cat_fare)
data.drop(['Fare'], axis=1, inplace=True)

# Next split the Name and Title 
data['Title'] = data['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
data.drop(['Name'], axis=1, inplace=True)
data['Title_enc'] = np.nan
data['Title_enc'] = data['Title'].map(data.groupby('Title')['Survived'].mean())
data['Title_enc'].fillna(data['Survived'].mean())
#data.drop(['Title'], axis=1, inplace=True)

# We donot need the passenger's id
data.drop(['PassengerId'], axis=1, inplace=True)
 
# fill the age, embarked na row with mean
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna('S', inplace=True)
 
# get the catagorical variable result
dataf = pd.get_dummies(data.drop(['Title'],axis=1), 
                       columns=['Sex','Embarked','Price_range'], 
                       prefix=['Sex','Embarked','Price_range'])

X = dataf.drop(['Survived'], axis=1)
y = dataf['Survived']
 
# cross validation
kf = StratifiedKFold(5, shuffle=False, random_state=1)
xgb = XGBClassifier(n_estimator=900, learning_rate = 0.1, random_state=1)
# rfc = RandomForestClassifier(random_state=1)
self_try = pd.DataFrame() 
self_try['True'] = y
 
# split the data into training and validation part

# Using only the cross validation

#for tr_ind, val_ind in kf.split(X,y):
#    X_tr = X.iloc[tr_ind]
#    X_val = X.iloc[val_ind]
#    y_tr = y.iloc[tr_ind]
#    y_val = y.iloc[val_ind]
#    xgb.fit(X_tr, y_tr)
#    self_try.loc[val_ind, 'XGBoost_result'] =  xgb.predict(X_val)
    # self_try.loc[val_ind, 'RandomForest_result'] =  rfc.predict(X_val)
    # xgboost is more accurate, so we avoid using RandomForest 
    
# Using the grid search
#
#grid_para = {'n_estimator': [500, 550],
#             'max_depth': [5], 
#             'random_state': [0, 1]}
#
#gd_sr = GridSearchCV(estimator = XGBClassifier(),
#                     param_grid = grid_para,
#                     scoring = 'accuracy',
#                     cv = 5,
#                     n_jobs=-1) 
#
#gd_sr.fit(X, y)
#
#print('Best score for dataset:', gd_sr.best_score_) 
#print(gd_sr.best_params_)

# print(accuracy_score(self_try['True'], self_try['XGBoost_result']))

# Now we move to the actual training of the test


xgb_final = XGBClassifier(n_estimator=500, eta = 0.01, random_state=0,
                          max_depth = 3)
xgb_final.fit(X, y)

test = pd.read_csv("C:/Users/13745/Desktop/WT2/Work/test23.csv")
test.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)
test['Family_member'] = test['SibSp'] + test['Parch']
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test['Price_range'] = test['Fare'].apply(cat_fare)
test.drop(['Fare'], axis=1, inplace=True)
test['Title'] = test['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
test['Title_enc'] = test['Title'].map(data.groupby('Title')['Survived'].mean())
test['Title_enc'].fillna(data['Survived'].mean(), inplace=True)
test.drop(['Title'], axis=1, inplace=True)
test.drop(['Name'], axis=1, inplace=True)
test.set_index('PassengerId', inplace=True)
test['Age'].fillna(data['Age'].mean(), inplace=True)
test['Embarked'].fillna('S', inplace=True)

testf = pd.get_dummies(test, columns=['Sex','Embarked','Price_range'], 
                               prefix=['Sex','Embarked','Price_range'])

pred_df = pd.DataFrame()
pred_df['PassengerId'] = test.index

pred_df['Survived'] = xgb_final.predict(testf)
pred_df.to_csv('grid_search23.csv', index=False)