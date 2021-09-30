import estimators as e
import numpy as np
from utils import generate_data_dirichlet, split_train_test

Ns = 1
alpha = 2
N = 40
Ttrain = 100
Ttest = 100
p = 10

Xall, Call, Uall, lambdas = generate_data_dirichlet(Ns=Ns, T=Ttrain+Ttest, 
                                                        alpha=alpha, N=N)

X, Ctrue = Xall[0], Call[0]
Xtrain, Xtest = split_train_test(X, train_fraction=Ttrain/(Ttrain+Ttest), standardize=True)

Ctrain = (Xtrain.T @ Xtrain) / Ttrain
res = e.fit_ConservativePCA(Xtrain, Xtest, p)




