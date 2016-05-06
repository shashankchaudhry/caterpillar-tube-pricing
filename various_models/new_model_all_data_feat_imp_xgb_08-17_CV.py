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
import random
import operator
from matplotlib import pylab as plt

random.seed(42)

def catToFreq(x, *args):
    freq_series = args[0]
    return(freq_series[x])
    
def ohe_multicolumn(data,columns,quant_cols,prefix,quant_flag):
    col_size=data.shape[0];
    raw_col=pd.DataFrame(data[columns[0]]);
    raw_col.columns=['Test'];
    for i in columns[1:]:
        temp_column=pd.DataFrame(data[i]);
        temp_column.columns=['Test'];
        raw_col=raw_col.append(temp_column,ignore_index=True);
    ohe_columns=pd.get_dummies(raw_col['Test']);
    ohe_column_names=ohe_columns.columns.values;
    ohe_column_names=[prefix + x for x in ohe_column_names];
    ohe_values=ohe_columns.values;
#   neglect the ' ' value 
    result_ohe=np.zeros([col_size,len(ohe_column_names)-1]);
    
    for j in range(1,len(columns)+1):
        if quant_flag==1:
            quant_vals=data[quant_cols[j-1]].values;
            result_ohe=result_ohe+np.multiply(quant_vals.T,ohe_values[(j-1)*col_size:j*col_size,1:].T).T;
        else:
            result_ohe=result_ohe+ohe_values[(j-1)*col_size:j*col_size,1:];
    print('done')
    print(data.columns.values)
    new_df=pd.DataFrame(result_ohe);
    new_df.columns=ohe_column_names[1:]
    print(new_df.columns.values)
    data = data.set_index(pd.Series(range(col_size)))
    data=pd.concat([data,new_df],axis=1);
    ohe_column_names.pop(0)
    return (data,ohe_column_names);
    
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
    
def calcTVTransform(train, in_col, out_col, n_fold, train_inds, test_inds):
    mean0 = train.ix[train_inds, out_col].mean()
    print("mean0: %f" % mean0)
    train['_key1'] = train[in_col].astype('category').values.codes
    sub_train = train.ix[train_inds, ['_key1', out_col]]
    grp1 = sub_train.groupby(['_key1'])
    sum1 = grp1[out_col].aggregate(np.sum)
    cnt1 = grp1[out_col].aggregate(np.size)
    v_codes = train.ix[test_inds, '_key1']
    _sum = sum1[v_codes].values
    _cnt = cnt1[v_codes].values
    _cnt[np.isnan(_sum)] = 0    
    _sum[np.isnan(_sum)] = 0
    
    r = {}
    r['exp'] = (_sum + n_fold * mean0)/(_cnt + n_fold)
    r['cnt'] = _cnt
    return(r)
    
def meanEncodeColSingle(train, col_name, keepCnt=True):
    n_folds = 10
    n_rows = train.shape[0]
    indices = shuffle(range(n_rows))
    num_per_fold = np.floor(float(n_rows)/float(n_folds))
    train['mean_' + col_name] = 0
    if keepCnt:
        train['count_' + col_name] = 0
    for i,fold in enumerate(range(n_folds)):
        if i < (n_folds - 1):
            l_val = fold * num_per_fold
            r_val = (fold+1) * num_per_fold
        else:
            l_val = fold * num_per_fold
            r_val = n_rows
        test_indices = indices[l_val:r_val]
        train_indices = indices[0:l_val] + indices[r_val:]
        result = calcTVTransform(train, col_name, 'cost', n_folds, train_indices, test_indices)
        train.ix[test_indices,'mean_' + col_name] = result['exp']
        if keepCnt:
            train.ix[test_indices,'count_' + col_name] = result['cnt']
    return(train)
    
def countEncodeColSingle(train, col_name):
    unique_vals = train[col_name].unique()
    train['count_' + col_name]=0
    for v in unique_vals:
        count = train.ix[train[col_name] == v,col_name].shape[0]
        train.ix[train[col_name] == v,'count_' + col_name] = count
    return(train)
    
def create_feature_map(features):
    outfile = open('./data/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def simpleMeanEncodeColSingle(train, col_name, out_col_name):
    m = train.groupby([col_name])[out_col_name].mean()
    train[col_name].replace(m,inplace=True)
    return(train)
    
row_num = 0

print("load data...")

train_data = pd.read_csv('./data/competition_data/train_set.csv', parse_dates=[2,])
tube_data = pd.read_csv('./data/competition_data/tube.csv')
tube_data = tube_data.set_index('tube_assembly_id')
bom_data = pd.read_csv('./data/competition_data/bill_of_materials.csv')
bom_data = bom_data.set_index('tube_assembly_id')
spec_data = pd.read_csv('./data/competition_data/specs.csv')
spec_data = spec_data.set_index('tube_assembly_id')
tube_end_form_data = pd.read_csv('./data/competition_data/tube_end_form.csv')
tube_end_form_data = tube_end_form_data.set_index('end_form_id')

# collect all component ids and their weights
comp_adaptor=pd.read_csv('./data/competition_data/comp_adaptor.csv', usecols=['component_id','component_type_id','weight']);
comp_boss=pd.read_csv('./data/competition_data/comp_boss.csv', usecols=['component_id','component_type_id','weight']);
comp_elbow=pd.read_csv('./data/competition_data/comp_elbow.csv', usecols=['component_id','component_type_id','weight']);
comp_float=pd.read_csv('./data/competition_data/comp_float.csv', usecols=['component_id','component_type_id','weight']);
comp_hfl=pd.read_csv('./data/competition_data/comp_hfl.csv', usecols=['component_id','component_type_id','weight']);
comp_nut=pd.read_csv('./data/competition_data/comp_nut.csv', usecols=['component_id','component_type_id','weight']);
comp_other=pd.read_csv('./data/competition_data/comp_other.csv', usecols=['component_id','weight']);
comp_other['component_type_id']='CPT-9999';
comp_sleeve=pd.read_csv('./data/competition_data/comp_sleeve.csv', usecols=['component_id','component_type_id','weight']);
comp_straight=pd.read_csv('./data/competition_data/comp_straight.csv', usecols=['component_id','component_type_id','weight']);
comp_threaded=pd.read_csv('./data/competition_data/comp_threaded.csv', usecols=['component_id','component_type_id','weight']);
comp_tee=pd.read_csv('./data/competition_data/comp_tee.csv', usecols=['component_id','component_type_id','weight']);
comp_all=(comp_adaptor
          .append(comp_boss)
          .append(comp_elbow)
          .append(comp_float)
          .append(comp_hfl)
          .append(comp_nut)
          .append(comp_other)
          .append(comp_sleeve)
          .append(comp_straight)
          .append(comp_threaded)
          .append(comp_tee));
comp_wts_all=comp_all.append(pd.DataFrame({'component_id':['9999'],'component_type_id':['CPT-9999'],'weight':[0.000001]}))
comp_wts_all=comp_wts_all.set_index(['component_id']);

print('initial train shape ' + str(train_data.shape))

data = train_data
print('initial data shape ' + str(data.shape))

# join with other tables
data = data.join(tube_data, on=['tube_assembly_id'], how='left')
data = data.join(bom_data, on=['tube_assembly_id'], how='left')
data = data.join(spec_data, on=['tube_assembly_id'], how='left')
data = data.join(tube_end_form_data, on=['end_a'], how='left')
data = data.join(tube_end_form_data, on=['end_x'], how='left', rsuffix='_1')
print('data shape after join ' + str(data.shape))

# extract year, month and week etc. 
data['year'] = data.quote_date.dt.year
data['month'] = data.quote_date.dt.month
data['cum_month'] = data.quote_date.dt.year * 12.0 + data.quote_date.dt.month
data['week'] = data.quote_date.dt.dayofyear
data['week'] = data['week'].apply(lambda x: np.floor((x-1.0)/7.0) + 1.0)
data['dayofweek'] = data.quote_date.dt.dayofweek
data['dayofyear'] = data.quote_date.dt.dayofyear
data['quarter'] = data.quote_date.dt.quarter
data['day'] = data.quote_date.dt.day

# drop quote date
data = data.drop(['quote_date'], axis = 1)
print('data shape after dropping quote_date ' + str(data.shape))

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
data['volume_full'] = (data['diameter'].apply(lambda x: (x**2)/4.0)) * data['length']
                            
# encode comp data and spec data
compid_columns=['component_id_1','component_id_2','component_id_3','component_id_4','component_id_5','component_id_6','component_id_7','component_id_8'];
quant_columns=['quantity_1','quantity_2','quantity_3','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8'];
spec_columns=['spec1','spec2','spec3','spec4','spec5','spec6','spec7','spec8','spec9','spec10'];
(data,ohe_col_names)=ohe_multicolumn(data,compid_columns,quant_columns,'comp_',1)
print('shape after adding comp_id ' + str(data.shape)) 
(data,spec_col_names)=ohe_multicolumn(data,spec_columns,quant_columns,'spec_',0)
print('shape after adding specs ' + str(data.shape)) 
#data=data.drop(compid_columns,axis=1)
#data=data.drop(spec_columns,axis=1)
print('shape after dropping comp_id,spec, (still keeping quantity) ' + str(data.shape)) 

# encoding the weight
data['weight'] = 0
for col_name in ohe_col_names[2:]:
    comp_id = str(col_name).split('_')[1]
    comp_wt = comp_wts_all.ix[comp_id]['weight']
    add_wt = data[col_name].multiply(comp_wt)
    data['weight'] = data['weight'].add(add_wt)
print('shape after encoding weight ' + str(data.shape))

# drop one hot encoded cols:
data=data.drop(ohe_col_names,axis=1)
data=data.drop(spec_col_names,axis=1)
print('shape after dropping ohe cols ' + str(data.shape))

# count encode tube_assembly_id
data = countEncodeColSingle(data, 'tube_assembly_id')
data=data.drop(['tube_assembly_id'],axis=1)

# simple mean encode all object type vars
data_types = data.dtypes
all_cols = data.columns.values
for i,col_name in enumerate(all_cols):
    if data_types[i] == object:
        data = simpleMeanEncodeColSingle(data, col_name, 'cost')
print('shape after mean encoding ' + str(data.shape)) 

# separating train and test
train_data = data

print('train shape on separation ' + str(train_data.shape))

# remove ids and cost
label = np.log1p(np.array(train_data['cost']))

# drop id and cost
train_data = train_data.drop(['cost'], axis = 1)

print('train shape dropping cost ' + str(train_data.shape))

all_columns = train_data.columns.values
create_feature_map(all_columns)

train_data = np.array(train_data)
train_data = train_data.astype(float)
train_data = sparse.csr_matrix(train_data)

print('fitting to test data....')

def get_rmsle(y_pred, y_actual):
    diff = y_pred - y_actual
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

ss = cross_validation.ShuffleSplit(train_data.shape[0], n_iter=5, test_size=0.2, random_state=0)
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

print('starting cross validation ...')
mean_score,clf = get_cross_val_score(train_data, label, ss, 25)

xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 15, "seed": 42, "silent": 1,\
                "min_child_weight": 1, "gamma": 1, "subsample": 0.9, "colsample_bytree": 0.9}
num_rounds = 1000

dtrain = xgb.DMatrix(train_data, label=label)
gbdt = xgb.train(xgb_params, dtrain, num_rounds)

importance = gbdt.get_fscore(fmap='./data/xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('./data/feature_importance_xgb.png')