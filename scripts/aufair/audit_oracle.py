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

    for varname, varlist in groups.items():
        for var in varlist:
            test = data[features]
            test['group'] = var
            predict = learner.predict(np.array(test[features]))
            data['predict_{}_{}'.format(varname, var)] = predict

    return data

def binary_audit(data, features_audit, auditor, groups, seed=1):
    """
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
            Has a predict value for each possible protected value
            obtained from the audited learner

    features_audit: list
            features used to train auditor learner
    
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

    """
    results = {}

    for varname in groups.keys():
        
        protected1 = groups[varname][0]
        protected2 = groups[varname][1]

        data['difference'] = np.absolute(data['predict_{}_{}'.format(varname, protected1)] - \
                            data['predict_{}_{}'.format(varname, protected2)])

        #split the data into train/test using a 0.7/0.3 ratio
        np.random.seed(seed=seed)
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)

        train_x = np.array(train[features_audit])
        train_y = np.array(train['difference']).ravel()

        # train auditor
        auditor.fit(train_x, train_y)

        # predict on test set
        test_x = np.array(test[features_audit])
        test_y = np.array(test['difference'])
        predicted = auditor.predict(test_x)
        test['pred'] = predicted

        # compute accuracy
        accuracy = (predicted == test_y).astype('int32')
        positive_label = (predicted == 1).mean()
        results[varname] = np.inner(predicted, test_y) / predicted.sum()
    
    return results




        


