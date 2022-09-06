#%%
from genericpath import isfile
import pickle
from utils import get_dati_tommaso, get_dati_camcan,\
    split_train_test, merge_dicts, generate_data_dirichlet
import estimators
import numpy as np
from time import process_time
import ray
ray.init(num_cpus=12)
#%%
# Function executed by a single process
@ray.remote
def single_run(key, Xtrain, Xtest, Ctrue=None, Cbar=None):
    res = {}

    # if Ctrue is not None:
    #     tstart = process_time() 
    #     res["Oracle"] = estimators.fit_Oracle(Ctrue, Xtrain, Xtest)
    #     res["Oracle"]["time"] = process_time() - tstart
            
    # tstart = process_time() 
    # res["PCA_Minka"] = estimators.fit_PCA_Minka(Xtrain, Xtest)
    # res["PCA_Minka"]["time"] = process_time() - tstart
    
    # tstart = process_time() 
    # res["RIE"] = estimators.fit_RIE(Xtrain, Xtest)
    # res["RIE"]["time"] = process_time() - tstart

    for cv_scoring in ["likelihood", "completion_error"]: #"pseudolikelihood"
    #     tstart = process_time() 
    #     res[f"PCA_CV_{cv_scoring}"] = estimators.fit_PCA_CV(Xtrain, Xtest)
    #     res[f"PCA_CV_{cv_scoring}"]["time"] = process_time() - tstart
        
    #     tstart = process_time() 
    #     res[f"Shrink_CV_{cv_scoring}"] = estimators.fit_Shrinkage_CV(Xtrain, Xtest)
    #     res[f"Shrink_CV_{cv_scoring}"]["time"] = process_time() - tstart
        
    #     tstart = process_time() 
    #     res[f"RIE_CV_{cv_scoring}"] = estimators.fit_RIE_CV(Xtrain, Xtest)
    #     res[f"RIE_CV_{cv_scoring}"]["time"] = process_time() - tstart

    #     tstart = process_time() 
    #     res[f"ConservativePCA_CV_{cv_scoring}"] = estimators.fit_ConservativePCA_CV(Xtrain, Xtest)
    #     res[f"ConservativePCA_CV_{cv_scoring}"]["time"] = process_time() - tstart

        tstart = process_time() 
        r = estimators.fit_Shrinkage_biasedmatrix_CV(Xtrain, Xtest, Cbar, cv_scoring=cv_scoring)
        res[f"ShrinkGroup_CV_{cv_scoring}"] = r
        res[f"ShrinkGroup_CV_{cv_scoring}"]["time"] = process_time() - tstart


    # val_frac_GA = 1 / 6 # Comparable to the 6 folds used in cross-validation above
    # for cv_scoring in ["likelihood", "completion"]: #  "pseudolikelihood"
    #     for b in [True, False]:
    #         # name = "GA_bootstrapping={b}_stop={cv_scoring}"
    #         # tstart = process_time() 
    #         # res[name] = estimators.fit_GradientAscent(Xtrain, Xtest, 
    #         #                     bootstrapping=False, stop=cv_scoring, val_frac=val_frac_GA)
    #         # res[name]["time"] = process_time() - tstart
            
    #         name = f"GAW_bootstrapping={b}_stop={cv_scoring}"
    #         tstart = process_time() 
    #         res[name] = estimators.fit_GradientAscentWishart(Xtrain, Xtest, 
    #                             bootstrapping=b, stop=cv_scoring, val_frac=val_frac_GA)
    #         res[name]["time"] = process_time() - tstart

        
    # res["Lasso_CV"] = estimators.fit_GraphicalLasso_CV(Xtrain, Xtest)
    # res["FA_CV"]  = estimators.fit_FactorAnalysis_CV(Xtrain, Xtest)
    return key, res

def parallel_run_tommaso():
    # resfile = 'all_results_tommaso.pickle'
    resfile = 'results_ShrinkGroup_tommaso.pickle'
    Xall = get_dati_tommaso(standardize=True)
    train_fraction = 0.8
    
    futures = []
    for i, X in Xall.items():
        # Compute the corr mat C obtained flattening the training data from all patients
        Cbar = computeCbar(Xall, i, train_fraction=train_fraction)
        Xtrain, Xtest = split_train_test(X, train_fraction=train_fraction, standardize=True, seed=i)
        futures.append(single_run.remote(i, Xtrain, Xtest, Cbar=Cbar))
    wait_and_dump(futures, resfile)

def parallel_run_camcan():
    resfile = 'all_results_camcan.pickle'
    Xall = get_dati_camcan(standardize=True)
    train_fraction = 0.8
    
    futures = []
    for i, X in Xall.items():
        # Compute the corr mat C obtained flattening the training data from all patients
        Cbar = computeCbar(Xall, i, train_fraction=train_fraction)
        Xtrain, Xtest = split_train_test(X, train_fraction=train_fraction, standardize=True, seed=i)
        futures.append(single_run.remote(i, Xtrain, Xtest, Cbar=Cbar))
    wait_and_dump(futures, resfile)


def parallel_run_dirichelet(alpha=1, Ttrain=144):
    #I generate a single realisation of the synthetic dataset 
    Ns = 100
    Ttest = int(Ttrain * 20/80)  #small Ttest
    # Ttest = 1000    
    N = 116
    # resfile = 'all_results_dirichelet_finalv2_smallTtest.pickle'
    # resfile = 'results_FA_dirichelet_smallTtest.pickle'
    resfile = 'results_ShrinkGroup_dirichelet_smallTtest.pickle'
    # resfile = 'test.pickle'
    Xall, Call, Uall, lambdas = generate_data_dirichlet(Ns=Ns, T=Ttrain+Ttest, 
                                                        alpha=alpha, N=N)
    train_fraction = Ttrain/(Ttrain+Ttest)
        
    futures = []
    for i, X in Xall.items():
        key = (i, alpha, Ttrain, N)
        Ctrue = Call[i]
        Cbar = computeCbar(Xall, i, train_fraction=train_fraction)
        Xtrain, Xtest = split_train_test(X, train_fraction=train_fraction, standardize=True, seed=i)
        futures.append(single_run.remote(key, Xtrain, Xtest, Ctrue, Cbar=Cbar))
    wait_and_dump(futures, resfile, Call)


def wait_and_dump(futures, resfile, Call=None):
    remaining_refs = futures
    all_res = {}
    
    while len(remaining_refs) > 0:
        ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1, timeout=None)
        key, res = ray.get(ready_refs[0])
        if Call is not None:
            res["Ctrue"] = Call[key[0]]
        print(f"Run {key} completed, {len(remaining_refs)} remaining.")
        all_res[key] = res

    if isfile(resfile):
        with open(resfile, 'rb') as f:
            all_results = pickle.load(f)
    else:
        all_results = {}
    merge_dicts(all_results, all_res, keepb=True)

    with open(resfile, 'wb') as f:
        pickle.dump(all_results, f)

def computeCbar(Xall, idx, train_fraction=0.8):
    assert len(Xall) > 1
    T, N = Xall[idx].shape
    Cbar = np.zeros((N, N))

    for idx2, X2 in Xall.items():
        if idx2 == idx:
            continue
        Xtrain2, Xtest2 = split_train_test(X2, train_fraction=train_fraction,standardize=True,seed=idx2)
        Cbar += np.corrcoef(Xtrain2.T)

    return Cbar / (len(Xall) - 1)

if __name__ == "__main__":
    # parallel_run_tommaso()
    for alpha in [1., 1.5, 2., 2.5, 3., 3.5, 4]:
        for Ttrain in [144, 300, 1000, 2000]:
            parallel_run_dirichelet(alpha, Ttrain)
 