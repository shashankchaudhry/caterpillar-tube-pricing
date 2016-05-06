# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:50:34 2015

@author: schaud7
"""

import pandas as pd
import numpy as np
from sklearn import ensemble
import logging
import datetime
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import make_scorer
import pdb
import xgboost as xgb

row_num = 0

logging.basicConfig(filename='log_file.log',level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)

logging.info("started...")
REGEN_DATA_FLAG = False

if(REGEN_DATA_FLAG == True):
    train_data = pd.read_csv('./data/competition_data/train_set.csv', parse_dates=[2,])
    #train_data = train_data.set_index('tube_assembly_id')
    test_data = pd.read_csv('./data/competition_data/test_set.csv', parse_dates=[3,])
    #test_data = test_data.set_index('tube_assembly_id')
    #train_data['quote_date'] = train_data['quote_date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))
    #test_data['quote_date'] = test_data['quote_date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))
    # unit cost is just cost
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
    
    # one hot encoding for supplier
    train_row_num = len(train_data['supplier'].values)
    test_row_num = len(test_data['supplier'].values)
    concat_suppliers = pd.concat([train_data['supplier'],test_data['supplier']])
    ohe_supplier = pd.get_dummies(concat_suppliers)
    ohe_supplier_train = ohe_supplier[:train_row_num]
    ohe_supplier_test = ohe_supplier[train_row_num:]
    for name in ohe_supplier_train.columns.values:
        train_data[name] = ohe_supplier_train[name]
        test_data[name] = ohe_supplier_test[name]
        
    print('supplier done')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # add tube.csv
    tube_data = pd.read_csv('./data/competition_data/tube.csv')
    tube_data = tube_data.set_index('tube_assembly_id')
    
    # inner join with train and test
    train_data = train_data.join(tube_data, on=['tube_assembly_id'], how='inner')
    test_data = test_data.join(tube_data, on=['tube_assembly_id'], how='inner')
    
    # one hot encoding for material_id
    train_row_num = len(train_data['material_id'].values)
    test_row_num = len(test_data['material_id'].values)
    concat_materials = pd.concat([train_data['material_id'],test_data['material_id']])
    ohe_material = pd.get_dummies(concat_materials)
    ohe_material_train = ohe_material[:train_row_num]
    ohe_material_test = ohe_material[train_row_num:]
    for name in ohe_material_train.columns.values:
        train_data[name] = ohe_material_train[name]
        test_data[name] = ohe_material_test[name]
        
    train_data['volume'] = (4.0*train_data['wall']*train_data['diameter'] - \
                            train_data['wall'].apply(lambda x: 4.0 * x**2)) * \
                            train_data['length']
                            
    test_data['volume'] = (4.0*test_data['wall']*test_data['diameter'] - \
                            test_data['wall'].apply(lambda x: 4.0 * x**2)) * \
                            test_data['length']
                            
    # drop material id column
    train_data = train_data.drop(['material_id'], axis = 1)
    test_data = test_data.drop(['material_id'], axis = 1)
    
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
    
    # combine with tube_end_form
    tube_end_form_data = pd.read_csv('./data/competition_data/tube_end_form.csv')
    train_data['end_a'] = train_data['end_a'].apply(lambda x: '9999' if x == 'NONE' else x)
    train_data['end_x'] = train_data['end_x'].apply(lambda x: '9999' if x == 'NONE' else x)
    test_data['end_a'] = test_data['end_a'].apply(lambda x: '9999' if x == 'NONE' else x)
    test_data['end_x'] = test_data['end_x'].apply(lambda x: '9999' if x == 'NONE' else x)
    tube_end_form_data = tube_end_form_data.set_index('end_form_id')
    
    train_data = train_data.join(tube_end_form_data, on=['end_a'], how='inner', rsuffix='_a')
    test_data = test_data.join(tube_end_form_data, on=['end_a'], how='inner', rsuffix='_a')
    train_data = train_data.join(tube_end_form_data, on=['end_x'], how='inner', rsuffix='_x')
    test_data = test_data.join(tube_end_form_data, on=['end_x'], how='inner', rsuffix='_x')
    
    train_data['forming'] = train_data['forming'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    test_data['forming'] = test_data['forming'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    train_data['forming_x'] = train_data['forming_x'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    test_data['forming_x'] = test_data['forming_x'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    
    # drop end_a and end_x columns
    train_data = train_data.drop(['end_a', 'end_x'], axis = 1)
    test_data = test_data.drop(['end_a', 'end_x'], axis = 1)
    
    print('done with tube')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # add bill_of_materials.csv
    bom_data = pd.read_csv('./data/competition_data/bill_of_materials.csv')
    bom_data = bom_data.set_index('tube_assembly_id')
    
    # preprocess BOM
    comp_data = pd.read_csv('./data/competition_data/components.csv')
    master_list = comp_data['component_id'].values
    for li in master_list:
        bom_data[li] = 0
    cols = ['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6'\
            , 'component_id_7', 'component_id_8']
    qt_cols = ['quantity_1', 'quantity_2', 'quantity_3', 'quantity_4', 'quantity_5', \
                'quantity_6', 'quantity_7', 'quantity_8']
    for i,rnum in enumerate(range(bom_data['component_id_1'].shape[0])):
        if (i % 500 == 0):
            print(i)
        for j,cname in enumerate(cols):
            comp_name = bom_data[cname].values[i]
            comp_qty = bom_data[qt_cols[j]].values[i]
            if not type(comp_name) == float:
                bom_data.ix[i,comp_name] = comp_qty
                
    print('BOM stuff done')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # inner join with train and test
    train_data = train_data.join(bom_data, on=['tube_assembly_id'], how='inner')
    test_data = test_data.join(bom_data, on=['tube_assembly_id'], how='inner')
    
    print('BOM inner join done')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # drop unnecessary columns
    train_data = train_data.drop(cols, axis = 1)
    train_data = train_data.drop(qt_cols, axis = 1)
    test_data = test_data.drop(cols, axis = 1)
    test_data = test_data.drop(qt_cols, axis = 1)
    
    print('BOM unnecessary cols dropped')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # add specs.csv
    spec_data = pd.read_csv('./data/competition_data/specs.csv')
    spec_data = spec_data.set_index('tube_assembly_id')
    column_names = spec_data.columns.values
    for i,cname in enumerate(column_names):
        if i == 0:
            master_list = spec_data[cname].unique()
        else:
            master_list = np.concatenate((master_list,spec_data[cname].unique()))
    master_list = np.unique(master_list)
    master_list = np.delete(master_list, [0], axis=0)
    for li in master_list:
        spec_data[li] = 0
    for i,rnum in enumerate(range(spec_data['spec1'].shape[0])):
        if (i % 500 == 0):
            print(i)
        for j,cname in enumerate(column_names):
            spec_name = spec_data[cname].values[i]
            if not type(spec_name) == float:
                spec_data.ix[i,spec_name] = 1
                
    print('spec comp done')
                
    # inner join the specs and remove the extra cols
    train_data = train_data.join(spec_data, on=['tube_assembly_id'], how='inner', rsuffix='_sp')
    test_data = test_data.join(spec_data, on=['tube_assembly_id'], how='inner', rsuffix='_sp')
    print('spec join done')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    train_data = train_data.drop(column_names, axis = 1)
    test_data = test_data.drop(column_names, axis = 1)
    print('spec col drop done')
    print('train shape' + str(train_data.shape))
    print('test shape' + str(test_data.shape))
    
    # Generate Labels and drop them from training set
    label = np.log1p(np.array(train_data['cost']))
    test_quantity = test_data['quantity'].values
    test_quantity = test_quantity.astype(float)
    train_data = train_data.drop('cost', axis = 1)
    
    # dropping unwanted columns
    drop_cols = ['tube_assembly_id','supplier','quote_date']
    train_data = train_data.drop(drop_cols, axis = 1)
    test_data = test_data.drop(drop_cols, axis = 1)
    
    # save to folder
    train_data.to_pickle('./data/pickle/train_set.pkl')
    test_data.to_pickle('./data/pickle/test_set.pkl')
else:
    temp_train_data = pd.read_csv('./data/competition_data/train_set.csv',usecols = ['cost'])
    temp_train_data['cost'] = temp_train_data['cost'].apply(float)
    label = np.log1p(np.array(temp_train_data['cost']))
    temp_test_data = pd.read_csv('./data/competition_data/test_set.csv', usecols = ['id'])
    idx = temp_test_data.id.values.astype(int)
    train_data = pd.read_pickle('./data/pickle/train_set.pkl')
    test_data = pd.read_pickle('./data/pickle/test_set.pkl')

train_data_np = np.array(train_data)
test_data_np = np.array(test_data)

train_data_np = train_data_np.astype(float)
test_data_np = test_data_np.astype(float)

# getting random subset of train for eval
train_size = train_data_np.shape[0]
eval_size = np.ceil(train_size * 0.1)
eval_indices = np.random.choice(train_size, eval_size, replace=False)
train_indices = [x for x in range(train_size) if x not in eval_indices]

new_train_label = label[train_indices]
new_eval_label = label[eval_indices]
new_train_data = train_data_np[train_indices]
new_eval_data = train_data_np[eval_indices]

logging.info('fitting to test data....')
results = []
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.3
params["min_child_weight"] = 6
params["gamma"] = 2.0
params["scale_pos_weight"] = 1.0
params["max_depth"] = 10
params["subsample"] = 0.85
params["colsample_bytree"] = 0.75
#params["booster"] = "gblinear"

plst = list(params.items())
xgtrain = xgb.DMatrix(new_train_data, label=new_train_label)
xgeval = xgb.DMatrix(new_eval_data, label=new_eval_label)
xgtest = xgb.DMatrix(test_data_np)

evallist  = [(xgeval,'eval'), (xgtrain,'train')]

num_rounds = 300
model = xgb.train(plst, xgtrain, num_rounds,evallist)

# get predictions from the model, convert them and dump them!
preds = np.expm1(model.predict(xgtest))
print(str(preds.shape))
print(str(idx.shape))
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('./data/results/result_All_train_500_Estimators_08-01-15_sub3.csv', index=False)