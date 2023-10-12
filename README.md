# Covariance Estimators

This repo contains utilities for denoising empirical covariance matrices.
It has been used to produce the results for the paper
["Noise-cleaning the precision matrix of fMRI time series"](https://arxiv.org/abs/2302.02951).

## Usage

### Install Environment
```
conda env create -f conda_env.yml
conda activate covest
```

### Estimators
The module `estimators.py` contains convenience functions in the form
`fit_METHOD(Xtrain, Xtest, ...)` that returns a denoised covariance matrix.
In the example below, we report all the available methods.

```python
import estimators as es 
import numpy as np

nsamples = 200
nfeatures = 116

X = np.random.randn(nsamples, nfeatures)  # dummy dataset with true covariance I
Xtrain = X[:int(0.8*nsamples)]
Xtest = X[int(0.8*nsamples):]

res = es.fit_PCA(Xtrain, Xtest, ncomponents=10)
# res = es.fit_PCA_CV(Xtrain, Xtest, cv_scoring="likelihood", n_jobs=None)
# res = es.fit_PCA_Minka(Xtrain, Xtest)
# res = es.fit_ConservativePCA(Xtrain, Xtest, p=10)
# res = es.fit_ConservativePCA_CV(Xtrain, Xtest, cv_scoring="likelihood", n_jobs=None)
# res = es.fit_FactorAnalysis(Xtrain, Xtest)
# res = es.fit_FactorAnalysis_CV(Xtrain, Xtest, cv_scoring="likelihood", n_jobs=None)
# res = es.fit_GraphicalLasso(Xtrain, Xtest, alpha=0.1)
# res = es.fit_GraphicalLasso_CV(Xtrain, Xtest, n_jobs=None)
# res = es.fit_Shrinkage(Xtrain, Xtest, shrinkage=0.1)
# res = es.fit_Shrinkage_CV(Xtrain, Xtest, shrinkages=None,  cv_scoring="likelihood", n_jobs=None)
# res = es.fit_RIE(Xtrain, Xtest)
# res = es.fit_RIE_CV(Xtrain, Xtest, cv_scoring="likelihood", n_jobs=None)
# res = es.fit_GradientAscent(Xtrain, Xtest, bootstrapping=False, stop='completion', val_frac=0.1)
# res = es.fit_GradientAscentWishart(Xtrain, Xtest, bootstrapping=False, stop='completion', val_frac=0.1)

C = X.T @ X / nsamples # empirical covariance
Cclean = res["Cclean"] # cleaned covariance
```
