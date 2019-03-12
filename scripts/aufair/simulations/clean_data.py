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
print(len(test) + len(train))


# community and crimes
crime = pd.read_csv('https://raw.githubusercontent.com/sethneel/GerryFair/dev_branch/dataset/communities.csv')
crime['Y'] = 2 * (crime.ViolentCrimesPerPop == 1).astype('int32') - 1
crime['attr'] = 2 * (crime['racepctblack'] >= 0.2).astype('int32') - 1
crime.to_csv('..\\..\\..\\data\\communities_crime_clean.csv')

# law school
data = pd.read_csv('..\\..\\..\\data\\admissions_bar.csv')
data['lsat'] = data.lsat.apply(lambda x: x.replace(' ', ''))
data['ugpa'] = data.ugpa.apply(lambda x: x.replace(' ', ''))
data['race'] = data.race.apply(lambda x: x.replace(' ', ''))
data['fam_inc'] = data.fam_inc.apply(lambda x: x.replace(' ', ''))
data = data[data.lsat != '']
data = data[data.ugpa != '']
data = data[data.race != '']
data = data[data.fam_inc != '']
data.loc[data.pass_bar == ' ', 'pass_bar'] = '0'
data = data[data.pass_bar.isin(['1', '0'])]
data = data[data.gender.isin(['male', 'female'])]

# create categorical data for age_cat, sex, race and charge degree
data['lsat'] = data.lsat.astype(float)
data['zfygpa'] = data.zfygpa.astype('category').cat.codes
data['zgpa'] = data.zgpa.astype('category').cat.codes
data['ugpa'] = data.ugpa.astype(float)
data['fam_inc'] = data.fam_inc.astype(float)
data['gender'] = data.sex.astype('category').cat.codes
data['race'] = data.race.astype('category').cat.codes
data['dropout'] = data.dropout.astype('category').cat.codes
data['fulltime'] = data.fulltime.astype('category').cat.codes
data['cluster'] = data.cluster.astype('category').cat.codes

data['Y'] = 2 * (data.pass_bar == 1).astype('int32') - 1
data['attr'] = 2 * (data.black == 1)