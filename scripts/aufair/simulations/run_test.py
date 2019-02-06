import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from aufair.simulations import test1 as t1
from aufair.simulations import test2 as t2
from aufair.simulations import test3 as t3
from aufair.simulations.create_data import dataset
from aufair import auditing as ad
#from aufair.simulations import test_net as tn


# figure 1a: decison tree --varying data size and unfairness intensity
def figure1a():
    nu_max = 11
    nu_min = 0
    noise = 0.2
    alpha = 0.1
    n = 500000

    # auditor
    dt = DecisionTreeClassifier()
    
    # cross validation
    max_depth = [1, 2, 4, 6, 8]
    min_samples_leaf = [1, 2, 5, 10]
    p_grid ={'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    parameter_grid = {'cv': 5, 'parameter': p_grid, 'niter': 20}

    # nboot runs for different sampe size
    results_list = []
    for ntest in [1000, 5000]:
        results = t1.test_certifying(n, ntest, nu_min, nu_max, dt, nboot=100, 
                                    sigma_noise=noise, 
                                    parameter_grid=parameter_grid,
                                    alpha=alpha)
        results['size'] = ntest
        results_list.append(results)
    
    report1 = pd.concat(results_list, axis=0)
    report1.to_csv('../../../results/synth_exp_sample_size_var.csv')

#  figure 1b: decison tree --varying data size and sub-population size
def figure1b():
    nu_max = 11
    nu_min = 0
    noise = 0.2
    n = 500000
    ntest = 5000
    nboot = 5
    alpha = 0.1
    unbalance = 0.25

    # auditor
    dt = DecisionTreeClassifier(max_depth=2)
    rf = RandomForestClassifier(n_estimators=50, max_depth=2)

    max_depth = [1, 2, 4, 6, 8]
    min_samples_leaf = [1, 2, 5, 10]
    p_grid = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    parameter_grid_tree = {'cv': 5, 'parameter': p_grid, 'niter': 20}

    svm_rbf = SVC(kernel='rbf', C=0.5)
    svm_lin = SVC(kernel='linear', C=0.5)

    C = [0.1, 0.5, 1, 1.5]

    p_grid = {'C': C}
    parameter_grid_svm = {'cv': 5, 'parameter': p_grid, 'niter': 4}

    auditor_dict = {'dt': (dt, parameter_grid_tree), 'rf': (rf, parameter_grid_tree),
                    'svm_rbf': (svm_rbf, parameter_grid_svm), 'svm_linear': (svm_lin, parameter_grid_svm)}

    # nboot runs for different sampe size
    results_list = []
    for key in auditor_dict.keys():
        auditor = auditor_dict[key][0]
        print(key)
        p_validation = auditor_dict[key][1]
        results = t1.test_certifying(n, ntest, nu_min, nu_max, auditor, nboot=nboot,
                                    sigma_noise=noise, 
                                    #parameter_grid=p_validation,
                                    alpha=alpha,
                                    unbalance=unbalance,
                                    balancing='MMD_NET')
        results['auditor'] = key
        results_list.append(results)
    
    report = pd.concat(results_list, axis=0)
    report.to_csv('../../../results/synth_exp_auditor_2a.csv')

# figure 2a: comparing different concept class
def figure2a():  
    nu_max = 11
    nu_min = 1
    noise = 0.2
    n = 500000  
    ntest = 5000
    nboot = 10
    alpha = 0.05
    unbalance = 0.25

    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=50)

    max_depth = [1, 2, 4, 6, 8]
    min_samples_leaf = [1, 2, 5, 10]
    p_grid ={'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    parameter_grid_tree = {'cv': 5, 'parameter': p_grid, 'niter': 20}

    svm_rbf = SVC(kernel='rbf')
    svm_lin = SVC(kernel='linear')

    C = [0.1, 0.5, 1, 1.5]

    p_grid ={'C': C}
    parameter_grid_svm = {'cv': 5, 'parameter': p_grid, 'niter': 4}

    auditor_dict = {'dt': (dt, parameter_grid_tree), 'rf': (rf, parameter_grid_tree), 
                    'svm_rbf': (svm_rbf, parameter_grid_svm), 'svm_linear': (svm_lin, parameter_grid_svm)}
    features = ['x1', 'x2']

    dset = dataset(sigma_noise=noise, unbalance=unbalance, n=ntest)
    dset.make_data(n)
    dset.classify()

    results_list = []
    
    for nu in np.arange(nu_min, nu_max):
        data = dset.simulate_unfairness(nu, alpha)
        data = dset.get_y(data)
        for i in np.arange(nboot):
            train, test = dset.split_train_test(data)

            for key in auditor_dict.keys():

                res = pd.DataFrame()
                auditor = auditor_dict[key][0]
                p_validation = auditor_dict[key][1]

                detect = ad.detector(auditor)
                train_x = np.array(train[features])
                train_y = np.array(train['label']).ravel()
                train_weights = np.array(train.weight).ravel()

                detect.certify(train_x, train_y, train_weights, parameter_grid=p_validation)

                test_x =  np.array(test[features])
                test_y = np.array(test['label']).ravel()
                test_weights = np.array(test.weight).ravel()
                test_a = np.array(test['attr']).ravel()
                pred =  np.array(test['predict']).ravel()
                gamma, acc = detect.certificate(test_x, test_y, pred, test_a, test_weights)
                
                res.loc[key, 'gamma'] =  alpha * nu / 20
                res.loc[key, 'estimated_gamma'] = gamma
                res.loc[key, 'iter'] = i
                res.loc[key, 'nu'] = nu
                res.loc[key, 'bias'] = gamma - res.loc[key, 'gamma']
                results_list.append(res)

    results = pd.concat(results_list)
    results.index.name = 'auditor'
    results.reset_index(inplace=True)
    report = results.groupby(['gamma', 'auditor'])[['estimated_gamma', 'bias']].mean()
    report['gamma_deviation'] = results.groupby(['gamma', 'auditor']).estimated_gamma.var()
    report['gamma_deviation'] = np.sqrt(report['gamma_deviation'])

    report.to_csv('../../../results/synth_exp_auditor_2a.csv')

def figure2b():
    nu_max = 5
    nu_min = 4
    noise = 0.2
    n = 500000
    ntest = 5000
    max_depth = [1, 2, 4, 6, 8]
    n_estimators = [10, 20, 50]
    min_samples_leaf = [1, 2, 5, 10]
    
    p_grid ={'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    parameter_grid_tree = {'cv': 5, 'parameter': p_grid, 'niter': 20}
    
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()

    results_list = []
    results = t1.test_certifying(n, ntest, nu_min, nu_max, dt, nboot=100, 
                                    sigma_noise=noise, 
                                    parameter_grid=parameter_grid)
    results['size'] = 'cv'
    results_list.append(results)

    """
    results = t1.test_certifying(n, ntest, nu_min, nu_max, dt, nboot=100, 
                                    sigma_noise=noise)
    results['size'] = 'no_cv'
    results_list.append(results)
    """
    
    report1 = pd.concat(results_list, axis=0)
    report1.to_csv('../../../results/synth_exp_sample_size_cv_test2.csv')


# figure 2: same sample size but different auditors:
def figure1e():  
    nu_max = 11
    nu_min = 1
    noise = 0.2
    n = 500000  
    ntest = 2500

    C = [0.5, 1, 1.5, 2, 10] 
   
    p_grid ={'C': C}
    parameter_grid = {'cv': 5, 'parameter': p_grid, 'niter': 5}
    print(p_grid)

    rbf = SVC(kernel='rbf')
    lin = SVC(kernel='linear')
    poly = SVC(kernel='poly')
    #dt = DecisionTreeClassifier(max_depth= 4)
    
    auditor_dict = {'rbf': rbf, 'linear': lin, 'poly': poly}

    results_list = []
    for key in auditor_dict.keys():
        auditor = auditor_dict[key]
        results = t1.test_certifying(n, ntest, nu_min, nu_max, auditor, nboot=10, 
                                    sigma_noise=noise, parameter_grid=parameter_grid)
        results['auditor'] = key
        results_list.append(results)
    
    report2 = pd.concat(results_list, axis=0)
    report2.to_csv('../../../results/synth_exp_auditor_svm.csv')

# compare different tree
def figure1d():  
    nu_max = 11
    nu_min = 1
    noise = 0.2
    n = 500000  
    ntest = 5000
    nboot = 100
    alpha = 0.05
    unbalance = 0

    dt_5= DecisionTreeClassifier(max_depth= 5)
    rf = RandomForestClassifier(n_estimators=50, max_depth=2)

    auditor_dict = {'dt_5': dt_5, 'rf': rf}
    features = ['x1', 'x2']

    dset = dataset(sigma_noise=noise, unbalance=unbalance, n=ntest)
    dset.make_data(n)
    dset.classify()

    results_list = []
    
    for nu in np.arange(nu_min, nu_max):
        data = dset.simulate_unfairness(nu, alpha)
        data = dset.get_y(data)
        for i in np.arange(nboot):
            train, test = dset.split_train_test(data)

            for key in auditor_dict.keys():

                res = pd.DataFrame()
                auditor = auditor_dict[key]

                detect = ad.detector(auditor)
                train_x = np.array(train[features])
                train_y = np.array(train['label']).ravel()
                train_weights = np.array(train.weight).ravel()

                detect.certify(train_x, train_y, train_weights)

                test_x =  np.array(test[features])
                test_y = np.array(test['label']).ravel()
                test_weights = np.array(test.weight).ravel()
                test_a = np.array(test['attr']).ravel()
                pred =  np.array(test['predict']).ravel()
                gamma, acc = detect.certificate(test_x, test_y, pred, test_a, test_weights)
                
                res.loc[key, 'gamma'] =  alpha * nu / 20
                res.loc[key, 'estimated_gamma'] = gamma
                res.loc[key, 'iter'] = i
                res.loc[key, 'nu'] = nu
                res.loc[key, 'bias'] = gamma - res.loc[key, 'gamma']
                results_list.append(res)

    results = pd.concat(results_list)
    results.index.name = 'auditor'
    results.reset_index(inplace=True)
    report = results.groupby(['gamma', 'auditor'])[['estimated_gamma', 'bias']].mean()
    report['gamma_deviation'] = results.groupby(['gamma', 'auditor']).estimated_gamma.var()
    report['gamma_deviation'] = np.sqrt(report['gamma_deviation'])

    report.to_csv('../../../results/synth_exp_auditor_test.csv')



def figure2d():
    nu_max = 11
    nu_min = 1
    noise = 0.2
    n = 500000  
    ntest = 10000
    unbalance = 0.25
    nboot = 50

    auditor = DecisionTreeClassifier(max_depth= 5)
    balancing_methods = { 'Uniform': None, 'IPW': 'IPW', 'IPW_Q': 'IPW_Q', 'MMD': 'MMD'}

    results_list = []
    for key in balancing_methods.keys():
        mod = balancing_methods[key]
        results = t2.test_certifying(n, ntest, nu_min, nu_max, auditor, 
                                    nboot=nboot, sigma_noise=noise, 
                                    balancing=mod, unbalance=unbalance)
        results['balancing'] = key
        results_list.append(results)

    #add representation
    results = tn.test_certifying(n, ntest, nu_min, nu_max, 
                                    nboot=20, sigma_noise=noise, 
                                    balancing='MMD', unbalance=unbalance)
    results['balancing'] = 'MMD_DNN'
    results_list.append(results)
    
    report2 = pd.concat(results_list, axis=0)
    report2.to_csv('../../../results/synth_exp_unbalance_2a_rep.csv')

# figure 3: unbalance data
def figure3a():
    nu_max = 6
    nu_min = 5
    noise = 0.2
    n = 500000  
    ntest = 10000
    nboot = 1

    auditor = DecisionTreeClassifier(max_depth= 5)
    balancing_methods = {'Uniform': None, 'IS': 'IS', 'MMD_NET': 'MMD_NET'}

    results_list = []
    for key in balancing_methods.keys():
        mod = balancing_methods[key]
        for unbalance in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            print(unbalance)
            results = t1.test_certifying(n, ntest, nu_min, nu_max, auditor, 
                                    nboot=nboot, sigma_noise=noise, 
                                    balancing=mod, unbalance=unbalance,
                                    lw = 0.01)
            results['balancing'] = key
            results['unbalance'] = unbalance
            results_list.append(results)
    
    report = pd.concat(results_list, axis=0)
    report.to_csv('../../../results/synth_exp_unbalance_3a_test.csv')

def figure3b():
    nu_max = 11
    nu_min = 1
    noise = 0.2
    n = 500000  
    ntest = 10000
    nboot = 10
    unbalance = 0.25

    auditor = DecisionTreeClassifier(max_depth= 5)
    
    results_list = []
    for lw in [10**(-4), 10**(-3), 10**(-2), 10**(-1)]:
        print(lw)
        results = t2.test_certifying(n, ntest, nu_min, nu_max, auditor, 
                                    nboot=nboot, sigma_noise=noise, 
                                    balancing='MMD', unbalance=unbalance, lw=lw)
        results['lw'] = lw
        results_list.append(results)
    
    report = pd.concat(results_list, axis=0)
    report.to_csv('../../../results/synth_exp_unbalance_3a.csv')


def figure4():
    nu_max = 10
    nu_min = 0
    noise = 0.2
    n = 500000
    ntest = 10000
    nboot = 10
    alpha = 0.15
    unbalance = 0.0

    # auditor
    rf = RandomForestClassifier(n_estimators=50)
    dt = DecisionTreeClassifier(max_depth=3)

    max_depth = [1, 2, 4, 6, 8]
    min_samples_leaf = [1, 2, 5, 10]
    p_grid = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    parameter_grid_tree = {'cv': 5, 'parameter': p_grid, 'niter': 20}

    # nboot runs for different sampe size
    results_list = []
    for step in [0.015]:
        print(step)
        results = t3.test_certifying(n, ntest, nu_min, nu_max, dt, nboot=nboot,
                                     sigma_noise=noise,
                                    # parameter_grid=parameter_grid_tree,
                                     alpha=alpha,
                                     unbalance=unbalance,
                                     balancing='MMD_NET',
                                     stepsize=step)
        results['step'] = step
        results_list.append(results)

    report = pd.concat(results_list, axis=0)
    report.to_csv('../../../results/synth_exp_violation_delta.csv')

if __name__ == "__main__":
    figure1b()

