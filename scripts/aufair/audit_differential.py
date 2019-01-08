import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

pd.options.mode.chained_assignment = None 


def binary_audit(data, features_audit, auditor, groups, seed=1, M=1, fraction=0.5):


    results = {}

    for varname in groups.keys():
        
        # create adjacent datasets
        protected = groups[varname]
        d1 = data[data[varname] == protected]
        d2 = data[data[varname] != protected]
        data['label' ] = 0 
        frac_protected  = len(d1) / len(data)

        for m in np.arange(M):

            # create labels
            
            ind = np.random.choice(d1.index, int((1-fraction) * len(d1)), replace=False)
            sample = pd.concat([d1.loc[ind, :], d2])
            sample['predict_adj'] = sample['predict'] / (frac_protected * (1- fraction) + 1 - frac_protected)
            sample = data.join(sample[['predict_adj']], how='left')
            sample['predict_adj'] = sample['predict_adj'].fillna(0)
           
            sample['label'] = ((sample[varname] == 0).astype('int32')) * sample['predict'] - ((sample[varname]== 1).astype('int32'))*sample['predict']
            data['label'] = data['label'] + sample['label']

        data['label'] = data['label'] / M
        
        #split the data into train/test using a 0.7/0.3 ratio
        np.random.seed(seed=seed)
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)

        train_x = np.array(train[features_audit])
        train_y = np.array(train['label']).ravel()

        # train auditor
        auditor.fit(train_x, train_y)
    
        # predict on test set
        test_x = np.array(test[features_audit])
        test_y = np.array(test['label'])
        predicted = auditor.predict(test_x)
        test['predicted'] = predicted
    
        # compute accuracy
        indicator = (predicted + 1) / 2
        accuracy = np.inner(predicted, test_y) / predicted.shape[0]
        #accuracy = predicted[predicted == test_y].shape[0] / predicted.shape[0]
        results[varname] = np.abs(accuracy) 
    
        # identify violations
        results['violations_%s' %varname] = predicted[predicted >= 0.1 ].shape[0] / predicted.shape[0]

    return results, predicted, test

def get_violation(data, auditor, features_audit, gamma, seed=1):

    #split the data into train/test using a 0.7/0.3 ratio
    np.random.seed(seed=seed)
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    train_x = np.array(train[features_audit])
    train_y = np.array(train['label']).ravel()

    # train auditor
    auditor.fit(train_x, train_y)
    
    # predict on test set
    test_x = np.array(test[features_audit])
    test_y = np.array(test['label'])
    predicted = auditor.predict(test_x)
    test['predicted2'] = predicted
    
    # get the threshold so that the resulting indicator measures unfairness equal to gamma
    gamma_est = 0
    gamma = -1
    t = 0
    alpha = 0
    iter = 0

    while (gamma_est > gamma ) & (t < 1):
        #print(gamma_est)
        gamma =  gamma_est
        indicator = (np.abs(predicted) >= t).astype('int32')
        gamma_est = np.abs(np.inner((predicted[indicator == 1] + 1) / 2, test_y[indicator == 1]) / predicted[indicator == 1].shape[0])
        t += 0.075
        iter += 1

        alpha = predicted[np.abs(predicted) >= t].shape[0] / predicted.shape[0]
        print(alpha)
    print(gamma_est)
    return alpha, test, t

def get_violations_iter(data, auditor, features_audit, gamma, seed=1):
    
    gamma = 0
    gamma_est = 0
    t = 0.0
    data['predicted2'] = 1
    
    #split the data into train/test using a 0.7/0.3 ratio
    np.random.seed(seed=seed)
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    train0 = train.copy()
    test = data.drop(train.index)
    test0 = test.copy()

    while (gamma_est >= 1.005 * gamma) & ( t < 0.95):
        #split the data into train/test using a 0.7/0.3 ratio
        np.random.seed(seed=seed)
        if len(train0[np.abs(train0.predicted2) <= t]) > 0:
            ind = np.random.choice(train0[np.abs(train0.predicted2) <= t].index, int(len(train0[np.abs(train0.predicted2) <= t]) * 0.1), replace=True)
            train = pd.concat([train0[np.abs(train0.predicted2) >= t], train0.loc[ind, :]])
        else:
            train = train0[np.abs(train0.predicted2) >= t]
        
        train_x = np.array(train[features_audit])
        train_y = np.array(train['label']).ravel()

        # train auditor
        auditor.fit(train_x, train_y)
    
        # predict on test set
        test_x = np.array(test[features_audit])
        test_y = np.array(test['label'])
        predicted = auditor.predict(train_x)
        train['predicted2'] = predicted

        
        # compute violations
        indicator = (predicted + 1) / 2
        gamma = gamma_est
        gamma_est = np.abs(np.inner(indicator, train_y) / train_y.shape[0])
        print(gamma_est)
        t += 0.05

    predicted = auditor.predict(test_x)
    test['predicted2'] = predicted
    test = test[test.predicted2 >= t]

    indicator = (predicted + 1) / 2
    gamma_est = np.abs(np.inner(indicator, test_y) / test_y.shape[0])

    alpha = len(test) / len(test0)
    print(gamma_est)

    return alpha, t, test

def get_violations_boosting(data, auditor, features_audit, gamma, seed=1):

    gamma = 0
    gamma_est = 0.000001
    niter = 0
    t = 0.005
    
    #split the data into train/test using a 0.7/0.3 ratio
    np.random.seed(seed=seed)
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)
    test['predicted2'] = 0

    train1 = train
    train0 = train
    index0 = []
    index1 = []
    while  (gamma_est > gamma * 1.0000) & (len(train0) > 0.0005 * len(train)):

        train_x = np.array(train0[features_audit])
        train_y = np.array(train0['label']).ravel()
        
        # train auditor
        auditor.fit(train_x, train_y)
    
        # predict on train set
        predicted = auditor.predict(train_x)
        train0['predicted2'] = predicted
        

        # compute violations
        indicator = (predicted + 1) / 2 
        gamma = gamma_est
        gamma_est = np.abs(np.inner(indicator, train_y)) / train0.shape[0]

        train0 = train0[train0.predicted2 >= gamma_est + t]
        print(len(train0))
        
        # compute fair group
        """
        train1 = train0.loc[~train0.index.isin(index0), :]
        
        train_x = np.array(train1[features_audit])
        train_y = np.array(train1['outcome']).ravel()
        
        # train auditor
        auditor.fit(train_x, train_y)

        predicted = auditor.predict(train_x)
        train1['predicted2'] = predicted
        index1 = train1[np.abs(train1.predicted2) >= np.abs(train1.predicted2.max()) - t].index
       
        
        train0 = train0.loc[~train0.index.isin(index1), :]
        """
        print(gamma_est)
        
        
    
        niter += 1
    
    t = np.abs(train0.predicted2).min()
    print(t)
    
    test_x = np.array(test[features_audit])
    test_y = np.array(test['label'])
    test['predicted2'] = auditor.predict(test_x)
    test1 = test[np.abs(test.predicted2) >= t]
    indicator = (test1.predicted2 + 1) / 2 
    gamma_est = np.abs(np.inner(indicator, test1.label)) / len(test)

    alpha = len(train0) / len(train)
    

    return alpha, t, train, train0 

def  get_violations_weight(data, auditor, features_audit, epsilon, seed=1):

    #split the data into train/test using a 0.7/0.3 ratio
    np.random.seed(seed=seed)
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    # change weight
    train['weight0'] = train['weight']
    train.loc[train.label == -1, 'weight'] = train.loc[train.label == -1, 'weight0'] * (1 + epsilon)

    # train auditor
    train_x = np.array(train[features_audit])
    train_y = np.array(train['label']).ravel()
    weights = np.array(train.weight / train.weight.sum()).ravel()

    auditor.fit(train_x, train_y, sample_weight=weights)

    # prediction
    test_x = np.array(test[features_audit])
    test_y = np.array(test['label'])
    test['predicted2'] = auditor.predict(test_x)
    test_end = test[test.predicted2 == 1]
    indicator = (test_end.predicted2 + 1) / 2 
    gamma_est = np.abs(np.sum(indicator * test_end.weight / test_end.weight.sum() * test_end.label))
    alpha = (test[test.predicted2 == 1].weight / test.weight.sum()).sum()

    return gamma_est, alpha, test, test_end

def get_violation_weight_iter(data, auditor, features_audit, seed=1):

    epsilon = 0
    eta  = 1
    gamma = 0
    gamma_est = 0.000001
    niter = 0
    epsilon0 = 0

    while (niter < 250):
        #gamma = gamma_est
        gamma_est, alpha, _, _ = get_violations_weight(data, auditor, features_audit, epsilon, seed=seed)
        print(gamma_est)
        if np.isnan(gamma_est):
            break
        if alpha < 0.01:
            epsilon0 -= 0.01
            break
        if np.isnan(alpha):
            break
        #if gamma_est >= 0.95 * gamma:
        epsilon += 0.01
        if gamma_est > gamma:
            epsilon0 = epsilon
            gamma = gamma_est

        niter += 1

    
    gamma_est, alpha, test, test_end = get_violations_weight(data, auditor, features_audit, epsilon0, seed=seed)
    test_end['label'] = test_end.label * test_end.weight / test_end.weight.sum()
    print(gamma_est)
    print(alpha)
    #print(test_end.describe())

    return alpha, test, test_end


    
    



