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
from sklearn import cross_validation
from sklearn.metrics import make_scorer
from random import shuffle

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
    print('num cols for %s = %d' % (col_name,len(ohe_column.columns.values)-1))
    train_col_names = ohe_column.columns.values
    for i, name in enumerate(train_col_names):
        if i > 0:
            #print('col name: %s' % (col_name + '_' + name))
            train[col_name + '_' + name] = ohe_column[name]
    train = train.drop(col_name, axis = 1)
    return(train)
    
def meanEncodeColSingle(train, col_name):
    nfolds = 10
    nrows = train.shape[0]
    rows_list
    ohe_column = pd.get_dummies(train[col_name])
    print('num cols for %s = %d' % (col_name,len(ohe_column.columns.values)-1))
    train_col_names = ohe_column.columns.values
    for i, name in enumerate(train_col_names):
        if i > 0:
            #print('col name: %s' % (col_name + '_' + name))
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
tube_end_form_data = pd.read_csv('./data/competition_data/tube_end_form.csv')
tube_end_form_data = tube_end_form_data.set_index('end_form_id')

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
data = data.join(tube_end_form_data, on=['end_a'], how='left')
data = data.join(tube_end_form_data, on=['end_x'], how='left', rsuffix='_1')
print('data shape after join ' + str(data.shape))

# extract year, month and week and 
data['year'] = data.quote_date.dt.year
data['month'] = data.quote_date.dt.month
data['week'] = data.quote_date.dt.dayofyear
data['week'] = data['week'].apply(lambda x: np.floor(x-1.0/7.0) + 1.0)
data['dayofweek'] = data.quote_date.dt.dayofweek
data['dayofyear'] = data.quote_date.dt.dayofyear
data['quarter'] = data.quote_date.dt.quarter
data['day'] = data.quote_date.dt.day

# drop quote date and tube_assembly_id
data = data.drop(['quote_date','tube_assembly_id'], axis = 1)
print('data shape after dropping quote_date and tube_assemby_id ' + str(data.shape))

# delete all cols with more than 75% nas
cols_to_drop = []
num_rows_in_data = data.shape[0]
for i,col_name in enumerate(data.columns.values):
    num_nas_in_col = data[col_name].isnull().sum()
    if float(num_nas_in_col)/float(num_rows_in_data) >= 0.75:
        cols_to_drop.append(col_name)
data = data.drop(cols_to_drop, axis = 1)
print('cols dropped: ' + str(cols_to_drop))
print('data shape after dropping these cols ' + str(data.shape))

# converting NA in to '0' and '" "' for mode Matrix Generation
data_types = data.dtypes
for i,col_name in enumerate(data.columns.values):
    if data_types[i] == object:
        data[col_name] = data[col_name].apply(lambda x: ' ' if type(x) == float else x)
    else:
        data[col_name] = data[col_name].apply(lambda x: 0 if math.isnan(x) == True else x)
        
print('data shape after cleaning ' + str(data.shape))

data['volume'] = (4.0*data['wall']*data['diameter'] - \
                            data['wall'].apply(lambda x: 4.0 * x**2)) * \
                            data['length']  

## mean encode all object type vars
#data_types = data.dtypes
#all_cols = data.columns.values
#for i,col_name in enumerate(all_cols):
#    if data_types[i] == object:
#        data = meanEncodeColSingle(data, col_name)
#print('shape after mean encoding ' + str(data.shape))

# one hot encode all object type vars
data_types = data.dtypes
all_cols = data.columns.values
for i,col_name in enumerate(all_cols):
    if data_types[i] == object:
        data = oneHotEncodeColSingle(data, col_name)
print('shape after OHE ' + str(data.shape)) 

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

# add intercept
train_data['intercept'] = 1.
test_data['intercept'] = 1.

all_columns = train_data.columns.values
out_columns = ['intercept']
# add supplier
for col in all_columns:
    if col.startswith('supplier'):
        out_columns.append(col)
out_columns.append('annual_usage')
out_columns.append('min_order_quantity')
# add bracket pricing
for col in all_columns:
    if col.startswith('bracket'):
        out_columns.append(col)
out_columns.append('quantity')
# add material_id
for col in all_columns:
    if col.startswith('material'):
        out_columns.append(col)
out_columns.append('diameter')

out_columns.append('wall')
out_columns.append('length')
out_columns.append('volume')
out_columns.append('num_bends')
out_columns.append('bend_radius')
for col in all_columns:
    if col.startswith('end_a_1'):
        out_columns.append(col)
for col in all_columns:
    if col.startswith('end_a_2'):
        out_columns.append(col)
for col in all_columns:
    if col.startswith('end_x_1'):
        out_columns.append(col)
for col in all_columns:
    if col.startswith('end_x_2'):
        out_columns.append(col)
for col in all_columns:
    if col.startswith('end_a') and not col.startswith('end_a_1') and not col.startswith('end_a_2'):
        out_columns.append(col)
for col in all_columns:
    if col.startswith('end_x') and not col.startswith('end_x_1') and not col.startswith('end_x_2'):
        out_columns.append(col)
out_columns.append('num_boss')
out_columns.append('num_bracket')
out_columns.append('other')

for col in all_columns:
    if col.startswith('component_id_1'):
        out_columns.append(col)
out_columns.append('quantity_1')

for col in all_columns:
    if col.startswith('component_id_2'):
        out_columns.append(col)
out_columns.append('quantity_2')

for col in all_columns:
    if col.startswith('component_id_3'):
        out_columns.append(col)
#out_columns.append('quantity_3')

for col in all_columns:
    if col.startswith('component_id_4'):
        out_columns.append(col)
#out_columns.append('quantity_4')

for col in all_columns:
    if col.startswith('component_id_5'):
        out_columns.append(col)
#out_columns.append('quantity_5')

for col in all_columns:
    if col.startswith('component_id_6'):
        out_columns.append(col)
#out_columns.append('quantity_6')

for col in all_columns:
    if col.startswith('component_id_7'):
        out_columns.append(col)
#out_columns.append('quantity_7')

for col in all_columns:
    if col.startswith('component_id_8'):
        out_columns.append(col)
#out_columns.append('quantity_8')

for col in all_columns:
    if col.startswith('spec'):
        out_columns.append(col)
        
# forming
for col in all_columns:
    if col.startswith('forming'):
        out_columns.append(col)
        
out_columns.append('year')
out_columns.append('month')
out_columns.append('week')
out_columns.append('dayofweek')
out_columns.append('dayofyear')
out_columns.append('quarter')
out_columns.append('day')

train_data = train_data[out_columns]
test_data = test_data[out_columns]

print('train shape dropping id and cost ' + str(train_data.shape))
print('test shape dropping id and cost ' + str(test_data.shape))

train_data = np.array(train_data)
test_data = np.array(test_data)

train_data = train_data.astype(float)
test_data = test_data.astype(float)

train_data = sparse.csr_matrix(train_data)
test_data = sparse.csr_matrix(test_data)

print('fitting to test data....')

def get_rmsle(y_pred, y_actual):
    diff = y_pred - y_actual
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

ss = cross_validation.ShuffleSplit(train_data.shape[0], n_iter=5, test_size=0.2, random_state=0)
clf = xgb.XGBRegressor(n_estimators = 200, learning_rate= 0.5, max_depth=15,min_child_weight=1,\
                        gamma=1,subsample=0.9,colsample_bytree=0.9)
num_runs = 25

def get_cross_val_score(train, label, ss, num_runs):
    n_runs = num_runs
    print("Performing cross validation..")
    overall_mean = 0.
    for i in range(n_runs):
        clf = xgb.XGBRegressor(n_estimators = 200, seed=i, learning_rate= 0.5, max_depth=15,min_child_weight=1,\
                        gamma=1,subsample=0.9,colsample_bytree=0.9)
        cross_val_prediction = cross_validation.cross_val_score(clf, train, label, cv=ss, n_jobs=-1, scoring = make_scorer(get_rmsle, greater_is_better=False))
        mean_score = 0.
        for j in range(len(cross_val_prediction)):
           mean_score += cross_val_prediction[j]
        mean_score = mean_score / len(cross_val_prediction)
        overall_mean += mean_score
        print('%d Mean CV score is %f: ' % (i, mean_score))
    print('Overall mean: %f' % (overall_mean/n_runs))
    return((overall_mean/n_runs,clf))

logging.info('starting cross validation ...')
mean_score,clf = get_cross_val_score(train_data, label, ss, 25)

for i in range(num_runs):
    print('test run: %d' % i)
    clf = xgb.XGBRegressor(n_estimators = 200, seed=i, learning_rate= 0.5, max_depth=15,min_child_weight=1,\
                        gamma=1,subsample=0.9,colsample_bytree=0.9)
    clf.fit(train_data,label)
    # get predictions from the model
    temp_preds = clf.predict(test_data)
    for j in range(temp_preds.shape[0]):
        if temp_preds[j] < 0:
            temp_preds[j] = 0
    if i == 0:
        tot_preds = temp_preds
    else:
        tot_preds += temp_preds

preds = np.expm1(tot_preds/num_runs)

# transform -ve preds to 0
for i in range(preds.shape[0]):
    if preds[i] < 0:
        preds[i] = 0

print(str(preds.shape))
print(str(idx.shape))
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('./data/results/result_python_remove_nans_08-16-15_sub1.csv', index=False)