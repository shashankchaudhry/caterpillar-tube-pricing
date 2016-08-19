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

row_num = 0

#logging.basicConfig(filename='log_file.log',level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

logging.info("started...")
train_data = pd.read_csv('./data/competition_data/shuffled_train_data_first_500.csv')
#train_data = train_data.set_index('tube_assembly_id')
test_data = pd.read_csv('./data/competition_data/test_set.csv')
#test_data = test_data.set_index('tube_assembly_id')
train_data['quote_date'] = train_data['quote_date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))
test_data['quote_date'] = test_data['quote_date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))
train_data['unit_cost'] = train_data['cost'] / train_data['quantity'].apply(int)

# encoding annual usage (0 or not)/ min_order_quantity/ bracket_pricing
train_data['is_annual_usage'] = train_data['annual_usage'].apply(lambda x: 1 if int(x) != 0 else 0)
test_data['is_annual_usage'] = test_data['annual_usage'].apply(lambda x: 1 if int(x) != 0 else 0)
train_data['is_min_order_quantity'] = train_data['min_order_quantity'].apply(lambda x: 1 if int(x) != 0 else 0)
test_data['is_min_order_quantity'] = test_data['min_order_quantity'].apply(lambda x: 1 if int(x) != 0 else 0)
train_data['bracket_pricing'] = train_data['bracket_pricing'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
test_data['bracket_pricing'] = test_data['bracket_pricing'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

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

# Generate Labels and drop them from training set
label = np.array(train_data['unit_cost'])
test_quantity = test_data['quantity'].values
test_quantity = test_quantity.astype(float)

# dropping unwanted columns
column_names = test_data.columns.values
drop_cols = ['tube_assembly_id','supplier','quote_date','material_id','end_a_1x','end_a_2x','end_x_1x','end_x_2x',\
            'end_a','end_x','id']
new_cols = [x for x in column_names if x not in drop_cols]

train_data = train_data[new_cols]
test_data = test_data[new_cols]

train_data = np.array(train_data)
test_data = np.array(test_data)

train_data = train_data.astype(float)
test_data = test_data.astype(float)

logging.info('fitting to test data....')
results = []

clf = ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=1)
clf.fit(train_data, label)
logging.info('Feature Importances: ' + str(clf.feature_importances_))

for item_no in range(test_data.shape[0]):
    test_case = test_data[item_no]
    results.append(clf.predict(test_case) * test_quantity[item_no])
    
preds = np.asarray(results)
logging.info('writing file....')

# Write predictions to file
sample = pd.read_csv('./data/sample_submission.csv')
sample['cost'] = preds[:,0]
sample.to_csv('./data/results/sample_result_500_train_07-11-15.csv', index = False)