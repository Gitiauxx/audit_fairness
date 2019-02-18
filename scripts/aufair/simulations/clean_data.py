import pandas as pd
import numpy as np

# adult data
train = pd.read_csv('..\\..\\..\\data\\adult_income_dataset.csv')

# clean features
train['workclass'] = train['workclass'].astype('category').cat.codes
train['education'] = train['education'].astype('category').cat.codes
train['occupation'] = train['occupation'].astype('category').cat.codes
train['relationship'] = train['relationship'].astype('category').cat.codes
train['marital-status'] = train['marital-status'].astype('category').cat.codes
train['income'] = train['income_bracket'].astype('category').cat.codes
train['gender'] =  train['sex'].astype('category').cat.codes
train['srace'] =  train['race'].astype('category').cat.codes
train['income'] = -2 * train.income + 1

test = pd.read_csv('..\\..\\..\\data\\adult_income_test.csv')
# clean features
test['workclass'] = test['workclass'].astype('category').cat.codes
test['education'] = test['education'].astype('category').cat.codes
test['occupation'] = test['occupation'].astype('category').cat.codes
test['relationship'] = test['relationship'].astype('category').cat.codes
test['marital-status'] = test['marital-status'].astype('category').cat.codes
test['income'] = test['income_bracket'].astype('category').cat.codes
test['gender'] =  test['sex'].astype('category').cat.codes
test['srace'] =  test['race'].astype('category').cat.codes

# define outcome and attribute
train['Y'] = 2 * (train['income_bracket'] == ' <=50K').astype('int32') - 1
test['Y'] = 2 * (test['income_bracket'] == ' <=50K.').astype('int32') - 1
train['attr'] = 2 * (train.sex == ' Female').astype('int32') - 1
test['attr'] = 2 * (test.sex == ' Female').astype('int32') - 1

train.to_csv('..\\..\\..\\data\\adult_income_train_clean.csv')
test.to_csv('..\\..\\..\\data\\adult_income_testclean.csv')


