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
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from sklearn.metrics import make_scorer

row_num = 0

logging.basicConfig(filename='log_file.log',level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)

logging.info("started...")
REGEN_DATA_FLAG = False
ONE_HOT_ENCODING = True

if(REGEN_DATA_FLAG == True):
    train_data = pd.read_csv('./data/competition_data/train_set.csv', parse_dates=[2,])
    test_data = pd.read_csv('./data/competition_data/test_set.csv', parse_dates=[3,])
    train_data['cost'] = train_data['cost'].apply(float)
    
    # for quote date
    train_data['year'] = train_data.quote_date.dt.year
    train_data['month'] = train_data.quote_date.dt.month
    train_data['dayofyear'] = train_data.quote_date.dt.dayofyear
    train_data['dayofweek'] = train_data.quote_date.dt.dayofweek
    train_data['day'] = train_data.quote_date.dt.day
    
    test_data['year'] = test_data.quote_date.dt.year
    test_data['month'] = test_data.quote_date.dt.month
    test_data['dayofyear'] = test_data.quote_date.dt.dayofyear
    test_data['dayofweek'] = test_data.quote_date.dt.dayofweek
    test_data['day'] = test_data.quote_date.dt.day
    print('quote date done')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # encoding annual usage (0 or not)/ min_order_quantity/ bracket_pricing
    train_data['is_annual_usage'] = train_data['annual_usage'].apply(lambda x: 1 if int(x) != 0 else 0)
    test_data['is_annual_usage'] = test_data['annual_usage'].apply(lambda x: 1 if int(x) != 0 else 0)
    train_data['is_min_order_quantity'] = train_data['min_order_quantity'].apply(lambda x: 1 if int(x) != 0 else 0)
    test_data['is_min_order_quantity'] = test_data['min_order_quantity'].apply(lambda x: 1 if int(x) != 0 else 0)
    train_data['bracket_pricing'] = train_data['bracket_pricing'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    test_data['bracket_pricing'] = test_data['bracket_pricing'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    
    # test ids
    idx = test_data.id.values.astype(int)
    test_data = test_data.drop(['id'], axis = 1)
    
    def catToFreq(x, *args):
        freq_series = args[0]
        return(freq_series[x])
        
    def oneHotEncodeCol(train, test, col_name):
        tr_row_num = len(train[col_name].values)
        concat_column = pd.concat([train[col_name],test[col_name]])
        ohe_column = pd.get_dummies(concat_column)
        ohe_column_train = ohe_column[:tr_row_num]
        ohe_column_test = ohe_column[tr_row_num:]
        for name in ohe_column_train.columns.values:
            train[name] = ohe_column_train[name]
            test[name] = ohe_column_test[name]
        train = train.drop(col_name, axis = 1)
        test = test.drop(col_name, axis = 1)
        return((train,test))
        
    def oneHotEncodeColSingle(train, col_name):
        ohe_column = pd.get_dummies(train[col_name])
        for name in ohe_column.columns.values:
            train[name] = ohe_column[name]
        train = train.drop(col_name, axis = 1)
        return(train)
    
    # encoding for supplier
    if ONE_HOT_ENCODING:
        train_data,test_data = oneHotEncodeCol(train_data,test_data,'supplier')
    else:
        concat_suppliers = pd.concat([train_data['supplier'],test_data['supplier']])
        supplier_counts = concat_suppliers.value_counts(dropna=False)
        train_data['supplier'] = train_data['supplier'].apply(catToFreq, args=(supplier_counts,))
        test_data['supplier'] = test_data['supplier'].apply(catToFreq, args=(supplier_counts,))
        
    print('supplier done')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # add tube.csv
    tube_data = pd.read_csv('./data/competition_data/tube.csv')
    tube_data = tube_data.set_index('tube_assembly_id')
    
    # inner join with train and test
    train_data = train_data.join(tube_data, on=['tube_assembly_id'], how='inner')
    test_data = test_data.join(tube_data, on=['tube_assembly_id'], how='inner')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # encoding for material_id
    # remove NaNs
    train_data['material_id'] = train_data['material_id'].apply(lambda x: 'SP-Nan' if type(x) == float else x)
    test_data['material_id'] = test_data['material_id'].apply(lambda x: 'SP-Nan' if type(x) == float else x)
    
    if ONE_HOT_ENCODING:
        train_data,test_data = oneHotEncodeCol(train_data,test_data,'material_id')
    else:
        concat_materials = pd.concat([train_data['material_id'],test_data['material_id']])
        material_id_counts = concat_materials.value_counts(dropna=False)
        print(material_id_counts)
        train_data['material_id'] = train_data['material_id'].apply(catToFreq, args=(material_id_counts,))
        test_data['material_id'] = test_data['material_id'].apply(catToFreq, args=(material_id_counts,))
        
    train_data['volume'] = (4.0*train_data['wall']*train_data['diameter'] - \
                            train_data['wall'].apply(lambda x: 4.0 * x**2)) * \
                            train_data['length']
                            
    test_data['volume'] = (4.0*test_data['wall']*test_data['diameter'] - \
                            test_data['wall'].apply(lambda x: 4.0 * x**2)) * \
                            test_data['length']
                            
    print('material id done')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
                            
    # process a_1x, a_2x, x_1x, x_2x
    train_data['end_a_1x'] = train_data['end_a_1x'].apply(lambda x: 1 if x.lower() == 'y' else 0)
    test_data['end_a_1x'] = test_data['end_a_1x'].apply(lambda x: 1 if x.lower() == 'y' else 0)
    
    train_data['end_a_2x'] = train_data['end_a_2x'].apply(lambda x: 1 if x.lower() == 'y' else 0)
    test_data['end_a_2x'] = test_data['end_a_2x'].apply(lambda x: 1 if x.lower() == 'y' else 0)
    
    train_data['end_x_1x'] = train_data['end_x_1x'].apply(lambda x: 1 if x.lower() == 'y' else 0)
    test_data['end_x_1x'] = test_data['end_x_1x'].apply(lambda x: 1 if x.lower() == 'y' else 0)
    
    train_data['end_x_2x'] = train_data['end_x_2x'].apply(lambda x: 1 if x.lower() == 'y' else 0)
    test_data['end_x_2x'] = test_data['end_x_2x'].apply(lambda x: 1 if x.lower() == 'y' else 0)
    
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # combine with tube_end_form
    tube_end_form_data = pd.read_csv('./data/competition_data/tube_end_form.csv')
    # add col to tube_end_form :::: NONE -> No
    #train_data['end_a'] = train_data['end_a'].apply(lambda x: '9999' if x == 'NONE' else x)
    #train_data['end_x'] = train_data['end_x'].apply(lambda x: '9999' if x == 'NONE' else x)
    #test_data['end_a'] = test_data['end_a'].apply(lambda x: '9999' if x == 'NONE' else x)
    #test_data['end_x'] = test_data['end_x'].apply(lambda x: '9999' if x == 'NONE' else x)
    tube_end_form_data = tube_end_form_data.set_index('end_form_id')
    
    train_data = train_data.join(tube_end_form_data, on=['end_a'], how='inner', rsuffix='_a')
    test_data = test_data.join(tube_end_form_data, on=['end_a'], how='inner', rsuffix='_a')
    train_data = train_data.join(tube_end_form_data, on=['end_x'], how='inner', rsuffix='_x')
    test_data = test_data.join(tube_end_form_data, on=['end_x'], how='inner', rsuffix='_x')
    
    train_data['forming'] = train_data['forming'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    test_data['forming'] = test_data['forming'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    train_data['forming_x'] = train_data['forming_x'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    test_data['forming_x'] = test_data['forming_x'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    
    # one hot encode end_a and end_x columns
    if ONE_HOT_ENCODING:
        train_data,test_data = oneHotEncodeCol(train_data,test_data,'end_a')
        train_data,test_data = oneHotEncodeCol(train_data,test_data,'end_x')
    else:
        train_data = train_data.drop(['end_a', 'end_x'], axis = 1)
        test_data = test_data.drop(['end_a', 'end_x'], axis = 1)
    
    print('done with tube')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    if ONE_HOT_ENCODING:
        # add bill_of_materials.csv
        bom_data = pd.read_csv('./data/competition_data/bill_of_materials.csv')
        bom_data = bom_data.set_index('tube_assembly_id')
        
        cols = ['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6'\
                , 'component_id_7', 'component_id_8']
        qt_cols = ['quantity_1', 'quantity_2', 'quantity_3', 'quantity_4', 'quantity_5', \
                    'quantity_6', 'quantity_7', 'quantity_8']
        
        # remove na's from cols and qt_cols
        for i, col in enumerate(cols):
            bom_data[col] = bom_data[col].apply(lambda x: ('C-Nan-' + str(i)) if type(x) == float else x)
            
        for i, qt_col in enumerate(qt_cols):
            bom_data[qt_col] = bom_data[qt_col].apply(lambda x: 0 if math.isnan(x) == True else x)
            
        # one hot encode cols
        for i, col in enumerate(cols):
            bom_data = oneHotEncodeColSingle(bom_data,col)
                    
        print('BOM stuff done')
        print('train shape' + str(train_data.shape))
        print('test shape' + str(test_data.shape))
        
        # inner join with train and test
        train_data = train_data.join(bom_data, on=['tube_assembly_id'], how='inner', rsuffix='_bom')
        test_data = test_data.join(bom_data, on=['tube_assembly_id'], how='inner', rsuffix='_bom')
        
        print('BOM inner join done')
        print('train shape' + str(train_data.shape))
        print('test shape' + str(test_data.shape))
        
    else:
        # add bill_of_materials.csv
        bom_data = pd.read_csv('./data/competition_data/bill_of_materials.csv')
        bom_data = bom_data.set_index('tube_assembly_id')
        
        # delete the columns with 75% or more NAs
        print('initial bom shape' + str(bom_data.shape))
        col_names = bom_data.columns.values
        row_num = len(bom_data['component_id_1'].values)
        new_col_names = []
        for col_name in col_names:
            na_count = bom_data[col_name].isnull().sum()
            if float(na_count) / float(row_num) > 0.75:
                new_col_names.append(col_name)
        
        bom_data = bom_data.drop(new_col_names, axis = 1)
        print('bom shape' + str(bom_data.shape))
        
        cols = ['component_id_1', 'component_id_2']
        concat_cols = pd.Series([])
        for col in cols:
            bom_data[col] = bom_data[col].apply(lambda x: 'CP-Nan' if type(x) == float else x)
            concat_cols = pd.concat([concat_cols,bom_data[col]])
        comp_counts = concat_cols.value_counts(dropna=False)
        for col in cols:
            bom_data[col] = bom_data[col].apply(catToFreq, args=(comp_counts,))
        bom_data['sum_cols'] = 0
        for col in cols:
            bom_data['sum_cols'] += bom_data[col]
        print('BOM stuff done')
        print('train shape' + str(train_data.shape))
        print('test shape' + str(test_data.shape))
        
        # inner join with train and test
        train_data = train_data.join(bom_data, on=['tube_assembly_id'], how='inner')
        test_data = test_data.join(bom_data, on=['tube_assembly_id'], how='inner')
        
        print('BOM inner join done')
    
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    if ONE_HOT_ENCODING:
        # add specs.csv
        spec_data = pd.read_csv('./data/competition_data/specs.csv')
        spec_data = spec_data.set_index('tube_assembly_id')
        column_names = spec_data.columns.values
        for i,cname in enumerate(column_names):
            spec_data = oneHotEncodeColSingle(spec_data,cname)
                    
        print('spec comp done')
                    
        # inner join the specs
        train_data = train_data.join(spec_data, on=['tube_assembly_id'], how='inner', rsuffix='_sp')
        test_data = test_data.join(spec_data, on=['tube_assembly_id'], how='inner', rsuffix='_sp')
        print('spec join done')
        print('train shape' + str(train_data.shape))
        print('test shape' + str(test_data.shape))
    else:
        # add specs.csv
        spec_data = pd.read_csv('./data/competition_data/specs.csv')
        spec_data = spec_data.set_index('tube_assembly_id')
        
        # delete the columns with 75% or more NAs
        print('initial specs shape' + str(spec_data.shape))
        col_names = spec_data.columns.values
        row_num = len(spec_data['spec1'].values)
        new_col_names = []
        for col_name in col_names:
            na_count = spec_data[col_name].isnull().sum()
            if float(na_count) / float(row_num) > 0.75:
                new_col_names.append(col_name)
        
        spec_data = spec_data.drop(new_col_names, axis = 1)
        print('specs shape' + str(spec_data.shape))        
        
        column_names = spec_data.columns.values
        concat_cols = pd.Series([])
        for col in column_names:
            spec_data[col] = spec_data[col].apply(lambda x: 'Spec-Nan' if type(x) == float else x)
            concat_cols = pd.concat([concat_cols,spec_data[col]])
        spec_counts = concat_cols.value_counts(dropna=False)
        for col in column_names:
            spec_data[col] = spec_data[col].apply(catToFreq, args=(spec_counts,))
        spec_data['sum_cols'] = 0
        for col in column_names:
            spec_data['sum_cols'] += spec_data[col]
        print('Spec stuff done')
        print('train shape' + str(train_data.shape))
        print('test shape' + str(test_data.shape))
        # inner join the specs and remove the extra cols
        train_data = train_data.join(spec_data, on=['tube_assembly_id'], how='inner', rsuffix='_sp')
        test_data = test_data.join(spec_data, on=['tube_assembly_id'], how='inner', rsuffix='_sp')
        print('spec join done')
        print('train shape' + str(train_data.shape))
        print('test shape' + str(test_data.shape))
    
    # Generate Labels and drop them from training set
    label = np.log1p(np.array(train_data['cost']))
    test_quantity = test_data['quantity'].values
    test_quantity = test_quantity.astype(float)
    train_data = train_data.drop('cost', axis = 1)
    
    # dropping unwanted columns
    drop_cols = ['tube_assembly_id','quote_date']
    train_data = train_data.drop(drop_cols, axis = 1)
    test_data = test_data.drop(drop_cols, axis = 1)
    
    # save to folder
    train_data.to_pickle('./data/pickle/train_set_no_ohe_new.pkl')
    test_data.to_pickle('./data/pickle/test_set_no_ohe_new.pkl')
else:
    temp_train_data = pd.read_csv('./data/competition_data/train_set.csv',usecols = ['cost'])
    temp_train_data['cost'] = temp_train_data['cost'].apply(float)
    label = np.log1p(np.array(temp_train_data['cost']))
    temp_test_data = pd.read_csv('./data/competition_data/test_set.csv', usecols = ['id'])
    idx = temp_test_data.id.values.astype(int)
    train_data = pd.read_pickle('./data/pickle/train_set_no_ohe_new.pkl')
    test_data = pd.read_pickle('./data/pickle/test_set_no_ohe_new.pkl')

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
preds.to_csv('./data/results/result_All_train_500_Estimators_08-09-15_sub1.csv', index=False)