# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:50:34 2015

@author: schaud7
"""

from time import time
import pandas as pd
import numpy as np
import math
import xgboost as xgb
from random import shuffle
import random
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV
from operator import itemgetter
from sklearn.cross_validation import PredefinedSplit
from sklearn.metrics import make_scorer

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
    
    print(data.columns.values)
    new_df=pd.DataFrame(result_ohe);
    new_df.columns=ohe_column_names[1:]
    print(new_df.columns.values)
    data = data.reset_index(drop=True)
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
    
def labelEncodeColSingle(train, col_name):
    ohe_column = train[col_name].astype('category').values.codes
    print('label encoded col: %s' % col_name)
    train[col_name] = ohe_column
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
    m = train.groupby([col_name]).size()
    train['count_'  +col_name] = train[col_name].replace(m)
    return(train)
    
def simpleMeanEncodeColSingle(data, col_name, out_col_name, train_rows):
    print('simple mean encoding : %s' % col_name)
    train = data[:train_rows]
    m = train.groupby([col_name])[out_col_name].mean()
    n = data.groupby([col_name])[out_col_name].size()
    #p = train.groupby([col_name])[out_col_name].std()
    overall_mean = -999
    #overall_sdev = -999
    data['mean_' + col_name] = data[col_name].replace(m)
    data['count_'  +col_name] = data[col_name].replace(n)
    #data['sdev_' + col_name] = data[col_name].replace(p)
    temp_flag = data['mean_' + col_name].apply(lambda x: True if type(x) == str else False)
    data.ix[temp_flag,'mean_' + col_name] = overall_mean
    #data.ix[temp_flag,'sdev_' + col_name] = overall_sdev
    data = data.drop(col_name,axis=1)
    return(data)

def get_rmsle(y_pred, y_actual):
    diff = y_pred - y_actual
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)
    
def evalerror(preds, dtrain):
    labels = dtrain.get_label();
    n=len(labels);
    delta_error=(preds-labels);
    error_metric=np.sqrt((pow(np.linalg.norm(delta_error),2))/n);
    return 'error', error_metric
    
# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

"""
Takes: Pandas DataFrame : train_data
Returns: Pandas DataFrame : train_set, eval_set
"""
def getTrainAndEval(train_data):
    TEST_TRAIN_RATIO = 0.3
    unique_train_assem_ids = train_data['tube_assembly_id'].unique()
    tot_rows = train_data.shape[0]
    shuffle(unique_train_assem_ids)
    eval_ids = []
    count = 0
    for id in unique_train_assem_ids:
        eval_ids.append(id)
        count += train_data.ix[train_data['tube_assembly_id'] == id,:].shape[0]
        if (float(count)/float(tot_rows) > TEST_TRAIN_RATIO):
            break
    eval_set = train_data.ix[train_data['tube_assembly_id'].isin(eval_ids),:]
    train_set = train_data.ix[~train_data['tube_assembly_id'].isin(eval_ids),:]
    print('made train and eval sets')
    print('train data shape ' + str(train_set.shape))
    print('eval data shape ' + str(eval_set.shape))
    return(train_set,eval_set,eval_ids)
    
"""
Takes: Pandas DataFrame : train_data, eval_data (cost should be removed and set to 0 to represent test set)
Returns: train_data (np array, or sparse array), eval_data(np array, or sparse array), 
        train_label, eval_ids, all_column_names
"""
def preprocessData(train_data, eval_data, comp_wts_all):
    n_train_rows = train_data.shape[0]
    data = pd.concat([train_data,eval_data])
    # fix screwed up index
    data = data.reset_index(drop=True)
    
    # extract year, month and week and 
    data['year'] = data.quote_date.dt.year
    data['month'] = data.quote_date.dt.month
    data['cum_month'] = data.quote_date.dt.year * 12.0 + data.quote_date.dt.month
    #data['week'] = data.quote_date.dt.dayofyear
    #data['week'] = data['week'].apply(lambda x: np.floor((x-1.0)/7.0) + 1.0)
    #data['dayofweek'] = data.quote_date.dt.dayofweek
    #data['dayofyear'] = data.quote_date.dt.dayofyear
    #data['quarter'] = data.quote_date.dt.quarter
    #data['day'] = data.quote_date.dt.day
    
    # drop quote date and tube_assembly_id
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
    #(data,spec_col_names)=ohe_multicolumn(data,spec_columns,quant_columns,'spec_',0)
    #print('shape after adding specs ' + str(data.shape)) 
    data=data.drop(compid_columns,axis=1)
    #data=data.drop(spec_columns,axis=1)
    print('shape after dropping comp_id,spec, (still keeping quantity) ' + str(data.shape)) 
    
    # encoding the weight
    data['weight'] = 0
    for i,col_name in enumerate(ohe_col_names[2:]):
        comp_id = str(col_name).split('_')[1]
        comp_wt = comp_wts_all.ix[comp_id]['weight']
        add_wt = data[col_name].multiply(comp_wt)
        data['weight'] = data['weight'].add(add_wt)

    print('shape after encoding weight ' + str(data.shape))
    
    # drop one hot encoded cols:
    #data=data.drop(ohe_col_names,axis=1)
    data=data.drop(spec_columns,axis=1)
    print('shape after dropping ohe cols ' + str(data.shape))
    
    # count encode tube_assembly_id
    data = countEncodeColSingle(data, 'tube_assembly_id')
    data=data.drop(['tube_assembly_id'],axis=1)
    
    # label encode all object type vars
    data_types = data.dtypes
    all_cols = data.columns.values
    for i,col_name in enumerate(all_cols):
        if data_types[i] == object:
            data = oneHotEncodeColSingle(data, col_name)
            #print('shape after mean encoding ' + str(data.shape))
    
    # separating train and eval
    train_data = data[:n_train_rows]
    eval_data = data[n_train_rows:]
    
    print('train shape on separation ' + str(train_data.shape))
    print('eval shape on separation ' + str(eval_data.shape))
    
    # remove ids and cost
    eval_ids = eval_data.id.values.astype(int)
    train_label = np.log1p(np.array(train_data['cost']))
    
    # drop id and cost
    train_data = train_data.drop(['id','cost'], axis = 1)
    eval_data = eval_data.drop(['id','cost'], axis = 1)
    
    print('train shape dropping id and cost ' + str(train_data.shape))
    print('test shape dropping id and cost ' + str(eval_data.shape))
    
    #for col in train_data.columns.values:
    #    print("col name: %s, num nans: %d" % (col,train_data[col].isnull().sum()))
    
    all_columns = train_data.columns.values
    
    train_data = np.array(train_data)
    eval_data = np.array(eval_data)
    
    train_data = train_data.astype(float)
    eval_data = eval_data.astype(float)
    
    #train_data = sparse.csr_matrix(train_data)
    #eval_data = sparse.csr_matrix(eval_data)
    
    return((train_data, eval_data, train_label, eval_ids, all_columns))
    
def evalModel(train_data, eval_data, train_labels, eval_labels, seed):
    joined_data = np.concatenate((train_data,eval_data),axis=0)
    joined_labels = np.concatenate((train_labels,eval_labels),axis=0)
    train_mask = np.zeros(train_data.shape[0]) - 1.0
    eval_mask = np.zeros(eval_data.shape[0])
    joined_mask = np.concatenate((train_mask,eval_mask),axis=0)
    ps = PredefinedSplit(test_fold=joined_mask)
    loss  = make_scorer(get_rmsle, greater_is_better=False)
    train_data = sparse.csr_matrix(train_data)
    eval_data = sparse.csr_matrix(eval_data)
    
    clf = RandomForestRegressor(random_state=seed, verbose=1)
    #clf.fit(train_data, train_labels)
    #preds = clf.predict(eval_data)
    #print(get_rmsle(eval_labels, preds))
    ## achieves 0.263
    
    # specify parameters and distributions to sample from
    param_dist = {"n_estimators": sp_randint(300, 800),
                  "max_depth": sp_randint(10, 50),
                  "max_features": ['auto','sqrt','log2'],
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11)}
    
    # run randomized search
    n_iter_search = 60
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=ps, scoring=loss,
                                       n_iter=n_iter_search,n_jobs=-1,pre_dispatch='n_jobs',verbose=2)
    
    start = time()
    random_search.fit(joined_data, joined_labels)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)
    
def buildModel(train_data, eval_data, train_labels,n_seeds=10):
    overall_preds = None
    for i in range(n_seeds):
        print('seed: %d' % i)
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.05
        params["min_child_weight"] = 6
        params["subsample"] = 0.7
        params["colsample_bytree"] = 0.6
        params["scale_pos_weight"] = 0.8
        params["silent"] = 1
        params["max_depth"] = 8
        params["max_delta_step"]=2
        plst = list(params.items())
        
        xgtrain = xgb.DMatrix(train_data,label=train_labels)
        xgeval = xgb.DMatrix(eval_data)
        model = xgb.train(plst, xgtrain, num_boost_round=1300, feval=evalerror)
        preds = model.predict(xgeval)
        preds = np.expm1(preds)
    
        # transform -ve preds to 0
        for i in range(preds.shape[0]):
            if preds[i] < 0:
                preds[i] = 0
        if overall_preds == None:
            overall_preds = preds
        else:
            overall_preds += preds
    overall_preds = overall_preds / n_seeds
    #new_preds = model.predict(xgeval,pred_leaf=True)
    new_preds = None
    return((model,preds,new_preds))
    
if __name__ == "__main__":
    row_num = 0
    
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
    print('weight nans: %d' % (comp_wts_all['weight'].isnull().sum()) )
    comp_wts_all = comp_wts_all.fillna(0.)
    print('weight nans: %d' % (comp_wts_all['weight'].isnull().sum()) )
    
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
    
    # get back train data
    (train_set,eval_set,eval_ids) = getTrainAndEval(data.loc[data.id < 0,:])
    eval_cost = np.log1p(np.array(eval_set['cost']))
    eval_set['cost'] = 0.
    
    print('preprocessing data...')
    (train_set, eval_set, train_label, eval_ids, all_columns) = preprocessData(train_set,eval_set,comp_wts_all)
    print('training data...')
    evalModel(train_set, eval_set, train_label, eval_cost,15)
#    (train_set, eval_set, train_label, eval_ids, all_columns) = preprocessData(data.loc[data.id < 0,:],data.loc[data.id > 0,:],comp_wts_all)
#    (model,preds,new_preds) = buildModel(train_set, eval_set, train_label)
#    
#    
#    print(str(preds.shape))
#    print(str(eval_ids.shape))
#    preds = pd.DataFrame({"id": eval_ids, "cost": preds})
#    preds.to_csv('./data/results/result_one_hot_08-22-15_piece2.csv', index=False)
    """
    Results:Model with rank: 1
Mean validation score: -0.266 (std: 0.000)
Parameters: {'max_features': 'auto', 'min_samples_split': 7, 'n_estimators': 371, 'max_depth': 49, 'min_samples_leaf': 1}

Model with rank: 2
Mean validation score: -0.267 (std: 0.000)
Parameters: {'max_features': 'auto', 'min_samples_split': 10, 'n_estimators': 319, 'max_depth': 26, 'min_samples_leaf': 1}

Model with rank: 3
Mean validation score: -0.267 (std: 0.000)
Parameters: {'max_features': 'auto', 'min_samples_split': 10, 'n_estimators': 652, 'max_depth': 23, 'min_samples_leaf': 1}
    """
