# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:50:34 2015

@author: schaud7
"""

import pandas as pd
import numpy as np
from scipy import sparse
import logging
import math
import xgboost as xgb

def catToFreq(x, *args):
    freq_series = args[0]
    return(freq_series[x])
    
def oneHotEncodeCol(train, test, col_name):
    tr_row_num = len(train[col_name].values)
    concat_column = pd.concat([train[col_name],test[col_name]])
    ohe_column = pd.get_dummies(concat_column)
    print('num cols for %s = %d' % (col_name,len(ohe_column.columns.values)))
    ohe_column_train = ohe_column[:tr_row_num]
    ohe_column_test = ohe_column[tr_row_num:]
    for name in ohe_column_train.columns.values:
        train[col_name + '_' + name] = ohe_column_train[name]
        test[col_name + '_' + name] = ohe_column_test[name]
    train = train.drop(col_name, axis = 1)
    test = test.drop(col_name, axis = 1)
    return((train,test))
    
def oneHotEncodeColSingle(train, col_name):
    ohe_column = pd.get_dummies(train[col_name])
    print('num cols for %s = %d' % (col_name,len(ohe_column.columns.values)))
    for name in ohe_column.columns.values:
        train[col_name + '_' + name] = ohe_column[name]
    train = train.drop(col_name, axis = 1)
    return(train)

row_num = 0

logging.basicConfig(filename='log_file.log',level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)

print("load data...")

train_data = pd.read_csv('./data/competition_data/train_set.csv', parse_dates=[2,])
test_data = pd.read_csv('./data/competition_data/test_set.csv', parse_dates=[3,])
tube_data = pd.read_csv('./data/competition_data/tube.csv')
tube_data = tube_data.set_index('tube_assembly_id')
bom_data = pd.read_csv('./data/competition_data/bill_of_materials.csv')
bom_data = bom_data.set_index('tube_assembly_id')
spec_data = pd.read_csv('./data/competition_data/specs.csv')
spec_data = spec_data.set_index('tube_assembly_id')

# assign train ids and test cost
train_data['id'] = pd.Series([-x for x in range(1,train_data.shape[0]+1)])
test_data['cost'] = 0

print('initial train shape ' + str(train_data.shape))
print('initial test shape ' + str(test_data.shape))

data = train_data.append(test_data)
print('initial data shape ' + str(data.shape))

# join with other tables
data = data.join(tube_data, on=['tube_assembly_id'], how='left')
data = data.join(bom_data, on=['tube_assembly_id'], how='left')
data = data.join(spec_data, on=['tube_assembly_id'], how='left')
print('data shape after join ' + str(data.shape))

# extract year, month and week
data['year'] = data.quote_date.dt.year
data['month'] = data.quote_date.dt.month
data['week'] = data.quote_date.dt.weekofyear

# drop quote date and tube_assembly_id
data = data.drop(['quote_date','tube_assembly_id'], axis = 1)

# converting NA in to '0' and '" "' for mode Matrix Generation
data_types = data.dtypes
for i,col_name in enumerate(data.columns.values):
    if data_types[i] == object:
        data[col_name] = data[col_name].apply(lambda x: ' ' if type(x) == float else x)
    else:
        data[col_name] = data[col_name].apply(lambda x: 0 if math.isnan(x) == True else x)
        
print('data shape after cleaning ' + str(data.shape))

# separating train and test
train_data = data.loc[data.id < 0,:]
test_data = data.loc[data.id > 0,:]

print('train shape on separation ' + str(train_data.shape))
print('test shape on separation ' + str(test_data.shape))

# remove ids and cost
idx = test_data.id.values.astype(int)
label = np.log1p(np.array(train_data['cost']))

# drop id and cost
train_data = train_data.drop(['id','cost'], axis = 1)
test_data = test_data.drop(['id','cost'], axis = 1)

print('train shape dropping id and cost ' + str(train_data.shape))
print('test shape dropping id and cost ' + str(test_data.shape))

# one hot encode all object type vars
data_types = train_data.dtypes
train_cols = train_data.columns.values
for i,col_name in enumerate(train_cols):
    if data_types[i] == object:
        train_data, test_data = oneHotEncodeCol(train_data, test_data, col_name)
print('train shape after OHE ' + str(train_data.shape))   
print('test shape after OHE ' + str(test_data.shape))
        


train_data = np.array(train_data)
test_data = np.array(test_data)

train_data = train_data.astype(float)
test_data = test_data.astype(float)

train_data = sparse.csr_matrix(train_data)
test_data = sparse.csr_matrix(test_data)

print('fitting to test data....')

n_runs = 50
for i in range(n_runs):
    print('test run: %d' % i)
    clf = xgb.XGBRegressor(n_estimators = 200, seed = 42, learning_rate= 0.5, max_depth=15,min_child_weight=1,\
                        gamma=1,subsample=0.9,colsample_bytree=0.9)
    clf.fit(train_data,label)
    # get predictions from the model
    temp_preds = clf.predict(test_data)
    if i == 0:
        tot_preds = temp_preds
    else:
        tot_preds += temp_preds

preds = np.expm1(tot_preds/n_runs)

# transform -ve preds to 0
for i in range(preds.shape[0]):
    if preds[i] < 0:
        preds[i] = 0

print(str(preds.shape))
print(str(idx.shape))
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('./data/results/result_python_R_replica_08-09-15_sub1.csv', index=False)