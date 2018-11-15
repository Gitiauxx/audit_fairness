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

    group: dictionary
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
        data[ 'predict_{}_{}'.format(varname, varvalue)]

    return data

def 
