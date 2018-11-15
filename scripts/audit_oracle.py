import numpy as np
import pandas as pd


def oracle_access(data, features, learner, groups):
    """
    This function use features values in features to obtain a target value
    from learner for each possible protected groups

    Parameters:
    ----------
    data: pandas table 
            dataframe with values for each feature

    features: list
            features used to train learner

    learner: a sklearn object with a predict attribute
            learner called by the oracle

    groups: dictionary
            keys are protected attribute (e.g. gender)
            values are protected values (e.g. male, female)

    Return:
    ----------
    data: pandas table
        dataframe with a target value for each protected value

    """

    for varname, varvalue in groups.items():
        test = data[features]
        test['group'] = varvalue
        predict = learner.predict(np.array(test))
        data['predict_{}_{}'.format(varname, varvalue)]

    return data

def binary_audit(data, features, features_audit, audited, 
                auditor, groups, seed=1):
    """"
    This function return a measure of fairness violation
    when the protected values are binary (e.g. male vs female)
    and oracle access to the audited learner. The unfairness metric
    is computed using an auditor learner. The labels are data.difference, the absolute
    difference between the target values from the audited learner
    for each protected group

    Parameters:
    ----------
    data: pandas table 
            dataframe with values for each feature

    features: list
            features used to train audited learner

    features_audit: list
            features used to train auditor learner

    audited: an object with a predict and fit attribute
            learner called by the oracle
    
    auditor: an object with a predict and fit attribute
            learner used to audit the audited learner
            Labels are data[difference]

    groups: dictionary
            keys are protected attribute (e.g. gender)
            values are protected values (e.g. male, female)
            only binary choices among protected values

    Return:
    ----------
    results: dictionary
        keys are protected attribute (e.g. gender)
        value is the accuracy of auditor when predicting data['difference']
        for the protected attribute  

    """"
    data = oracle_access(data, features, audited, groups)

    results = {}

    for varname, varvalue in groups.keys():
        
        protected_1 = groups[varname][0]
        protected2 = groups[varname][1]

        data['difference'] = np.absolute(data['predict_{}_{}'.format(varname, protected_1)] - \
                            data['predict_{}_{}'.format(varname, protected_2)]

        #split the data into train/test using a 0.7/0.3 ratio
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)

        train_x = np.array(train[features_audit])
        train_y = np.array(train['difference']).ravel()

        # train auditor
        auditor.fit(train_x, train_y)

        # predict on test set
        test_x = np.array(test[features_audit])
        test_y = np.array(test['difference'])
        predicted = auditor.predict()

        # compute accuracy
        accuracy = (predicted == train_y).astype('int32')
        result[varname] = accuracy
    
    return results


        


