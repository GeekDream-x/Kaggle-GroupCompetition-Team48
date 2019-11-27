#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:32:44 2019

"""
#import modules
import pandas as pd
import numpy as np
def create_count(df,feat):
    feat_count = df.groupby(feat).size().reset_index()
    feat_count.columns = [feat,'%s_count'%(feat)]
    df = df.merge(feat_count,how='left',on=[feat])
    return df
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

nrows = None
test = pd.read_csv("tcd-ml-1920-group-income-train.csv",nrows=nrows)
train = pd.read_csv("tcd-ml-1920-group-income-test.csv",nrows=nrows)
train = train.rename(columns={"Total Yearly Income [EUR]":'Income'})
test = test.rename(columns={"Total Yearly Income [EUR]":'Income'})
#replace missing values with mode
print(train.apply(lambda x:np.sum(x.isnull()))) 
train.fillna(value = {'Year of Record':train['Year of Record'].mode()[0],
            'Satisfation with employer':train['Satisfation with employer'].mode()[0],
            'Gender':train.Gender.mode()[0],
             'Profession':train.Profession.mode()[0],
             'University Degree':train['University Degree'].mode()[0],
             'Country':train.Country.mode()[0],
             'Hair Color':train['Hair Color'].mode()[0]},inplace=True)
print(train.apply(lambda x:np.sum(x.isnull()))) 

print(test.apply(lambda x:np.sum(x.isnull())))
test.fillna(value={'Year of Record':test['Year of Record'].mode()[0],
            'Satisfation with employer':test['Satisfation with employer'].mode()[0],
            'Gender':test.Gender.mode()[0],
             'Profession':test.Profession.mode()[0],
             'University Degree':test['University Degree'].mode()[0],
             'Hair Color':test['Hair Color'].mode()[0]},inplace=True)
print(test.apply(lambda x:np.sum(x.isnull())))
# #replacing 0s with categorical values
train['Housing Situation'] = train['Housing Situation'].replace(['0','nA' ], 'Unknown')   
train['Housing Situation'] = train['Housing Situation'].replace([' ' ], '')
train['University Degree'] = train['University Degree'].replace(['0','No'], 'Unknown')
train['Hair Color'] = train['Hair Color'].replace(['0','unknown'], 'Unknown')
train['Gender'] = train['Gender'].replace(['0','unknown'], 'Unknown')
train['Gender'] = train['Gender'].replace(['f'], 'female')
train['Work Experience in Current Job [years]'] = train['Work Experience in Current Job [years]'].replace(['#NUM!'], 'NaN')
train['Yearly Income in addition to Salary (e.g. Rental Income)']=train['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace("EUR", "")

train['Housing Situation'] = train['Housing Situation'].replace(['0','nA' ], 'Unknown')   
train['Housing Situation'] = train['Housing Situation'].replace([' ' ], '')
train['University Degree'] = train['University Degree'].replace(['0','No'], 'Unknown')
train['Hair Color'] = train['Hair Color'].replace(['0','unknown'], 'Unknown')
train['Gender'] = train['Gender'].replace(['0','unknown'], 'Unknown')
train['Gender'] = train['Gender'].replace(['f'], 'female')
train['Work Experience in Current Job [years]'] = train['Work Experience in Current Job [years]'].replace(['#NUM!'], 'NaN')
train['Yearly Income in addition to Salary (e.g. Rental Income)']=train['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace("EUR", "")

print(train.describe())
print(test.describe())
#Combine training and testing data
all_data = pd.concat([train,test],ignore_index=True)
columns = {'Work Experience in Current Job [years]':'Work Experience in Current Job years',
 'Body Height [cm]':'Body Height cm',
 'Yearly Income in addition to Salary (e.g. Rental Income)':'Yearly Income in addition to Salary e.g. Rental Income'}
all_data = all_data.rename(columns=columns)


train.head()
test.head()
all_data.head()
#Feature encoding
le = LabelEncoder()
for col in all_data.dtypes[all_data.dtypes == 'object'].index.tolist():
    all_data[col] = le.fit_transform(all_data[col].astype(str))
    
feat_col = [col for col in all_data.columns if col not in ['Instance','Income']]
print(feat_col)

for col in feat_col:
    print(col)
    all_data = create_count(all_data,col)
    
train = all_data[all_data['Income'].notnull()]
test = all_data[all_data['Income'].isnull()]
len(all_data)
train.head()

feat_col = [col for col in train.columns if col not in ['Instance','Income']]
feat_col
#Split the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train[feat_col],train['Income'],test_size=0.2
                                                 ,random_state=2019)
'done'
#Run Lightgbm 
params = {
         'num_trees':10000,
          'max_depth': 15,
          'learning_rate': 0.01,
          'bagging_freq':1,
          'bagging_fraction':1,
          "boosting": "gbdt",
          "bagging_seed": 11,
          'objective':'tweedie',
          "metric": 'mae',
          "verbosity": -1,
          
         }
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)
gbm = lgb.train(params, train_data, 100000, valid_sets = [train_data, test_data],verbose_eval=1000, early_stopping_rounds=1000)
pre_test1 = gbm.predict(x_test.values)
pre_test2 = gbm.predict(test[feat_col].values)
'done'
#Got the MAE score
MAE=mean_absolute_error(y_test.values,pre_test1)
print(MAE)

sub = pd.DataFrame()
sub['Instance'] = test['Instance'].tolist()
sub['Total Yearly Income [EUR]'] = pre_test2
sub.to_csv("submitty.csv",index=False)
sub.head()