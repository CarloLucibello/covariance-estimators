import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
# from sklearn.utils.testing import ignore_warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNetCV

import moduli_miguel as mm

### UTILITIES ##################################
def completion_error(dataset, myC, returnstd=True):
    myJ = np.linalg.inv(myC)
    d = np.diag(myJ)
    Jnod = myJ - np.diag(d) 
    def mu(x):
        return - (Jnod @ x) / d
    
    myvector = [np.mean(np.abs(x - mu(x))) for x in dataset]
    if returnstd:
        return np.mean(myvector), np.std(myvector,ddof=1)/np.sqrt(len(myvector))
    return np.mean(myvector)

def logpseudolikelihood(dataset, myC, returnstd=True):
    myJ = np.linalg.inv(myC)
    d = np.diag(myJ)
    Jnod = myJ - np.diag(d) 
    pslk1 = 0.5 * np.mean(np.log(d)) - 0.5 * np.log(2*np.pi) 
    
    def pslk(x):
        b = Jnod @ x
        pslk2 = -0.5*np.mean(b**2 / d) - 0.5*np.mean(x**2 * d) - np.mean(b*x)   
        return pslk1 + pslk2
    
    myvector = [pslk(x) for x in dataset]
    if returnstd:
        return np.mean(myvector), np.std(myvector,ddof=1)/np.sqrt(len(myvector))
    return np.mean(myvector)


#this function is to be used as in "3.3.1.3. Implementing your own scoring object" in 
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
# "without the make_scorer() factory"
def get_scoring_function(scoring=None):
    if scoring == "completion_error":
        def f(est, dataset, groundtruthtarget=None):
            myC = get_covariance(est)
            return -completion_error(dataset, myC, returnstd=False)
    elif scoring == "pseudolikelihood":
        def f(est, dataset, groundtruthtarget=None):
            myC = get_covariance(est)
            return logpseudolikelihood(dataset, myC, returnstd=False)
    elif scoring == "likelihood":
        def f(est, dataset, groundtruthtarget=None):
            myC = get_covariance(est)
            J = np.linalg.inv(myC)
            return mm.likelihood_set_fast(J, dataset)
    elif scoring is None:
        f = None
    return f

def get_covariance(est):
    if hasattr(est, "covariance_"):
        return est.covariance_
    else: 
        return est.get_covariance()


def compute_scores(est, Xtrain, Xtest, cross_val=False):
    res = {}
    res["train_likelihood"] = get_scoring_function("likelihood")(est, Xtrain)
    res["test_likelihood"] = get_scoring_function("likelihood")(est, Xtest)
    # res["train_likelihood"] = est.score(Xtrain)
    # res["test_likelihood"] = est.score(Xtest)
    res["train_completion_error"] = -get_scoring_function("completion_error")(est, Xtrain)
    res["test_completion_error"] = -get_scoring_function("completion_error")(est, Xtest)
    res["train_pseudolikelihood"] = get_scoring_function("pseudolikelihood")(est, Xtrain)
    res["test_pseudolikelihood"] = get_scoring_function("pseudolikelihood")(est, Xtest)
    res["Cclean"] = get_covariance(est)

    if cross_val:
        scoring = get_scoring_function("likelihood")
        res["cross_val_likelihood"] = np.mean(cross_val_score(est, Xtrain, cv=6, scoring=scoring))
        scoring = get_scoring_function("completion_error")
        res["cross_val_completion_error"] = np.mean(cross_val_score(est, Xtrain, cv=6, scoring=scoring))
        scoring = get_scoring_function("likelihood")
        res["cross_val_pseudolikelihood"] = np.mean(cross_val_score(est, Xtrain, cv=6, scoring=scoring))
        
    return res

############## ESTIMATORS ####################################

def fit_PCA_CV(Xtrain, Xtest, n_jobs=None, cv_scoring="likelihood"):
    scoring = get_scoring_function(cv_scoring)
    mycomponents = np.arange(2, Xtrain.shape[1], 1)
    cv_pca = GridSearchCV(PCA(svd_solver='full'), {'n_components': mycomponents}, cv=6,\
                     scoring=scoring, n_jobs=n_jobs)

    est = cv_pca.fit(Xtrain).best_estimator_
   
        #cv_p
    # print('this is a consistency test, and an illustration of GridSearchCV(). these numbers should coincide')
    # print(cv_pca.score(Xtrain))
    # print(cv_pca.score(Xtest))
    # print(PCA_completion_error_scoringfunction(est, Xtrain))
    # print(PCA_completion_error_scoringfunction(est, Xtest))
    # print(est.score(Xtrain))
    # print(est.score(Xtest))
    return compute_scores(est, Xtrain, Xtest)
    
def fit_PCA_Minka(Xtrain, Xtest):
    est = PCA(svd_solver='full',n_components='mle').fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
    

class RIE(BaseEstimator):
    def __init__(self, eta=1):
        self.Creg = None # come non-inizializzarla
        self.eta = eta
        
    def fit(self, X):
        self._fit(X)
        return self
    
    # https://www.researchgate.net/profile/Joel-Bun/publication/302339055_My_Beautiful_Laundrette_Cleaning_Correlation_Matrices_for_Portfolio_Optimization/links/572fabdf08aeb1c73d13a609/My-Beautiful-Laundrette-Cleaning-Correlation-Matrices-for-Portfolio-Optimization.pdf
    def _fit(self,Xtrain):
        T, N = np.shape(Xtrain)
        
        C_training = np.corrcoef(Xtrain.T) 
        eta = complex(0.,1.) * self.eta / np.sqrt(N)

        lambdas, E = np.linalg.eigh(C_training)
        lambdas = np.abs(np.real(lambdas))
        order = np.argsort(lambdas)[::-1]
        lambdas = lambdas[order]                     # reordering the eigvals
        E = E.T[order]                               # now each row of E is an eigvec
        #########
        q = N / T

        zs = lambdas - eta
        ss = np.array([np.sum(1 / (z - lambdas)) for z in zs]) / N
        ss += - 1 / (zs - lambdas) / N 
        xiRIE = lambdas / np.abs(1 - q + q*zs*ss)**2
        
        sigma2 = lambdas[-1] / (1 - np.sqrt(q))**2
        lambdaplus = lambdas[-1] * ((np.sqrt(q) + 1) / (-np.sqrt(q) + 1))**2

        # Stieltjes transform of the Marcenko-Pastur lambdas   
        gMPs = (zs + sigma2 *(q-1) - np.sqrt((zs-lambdas[-1])*(zs-lambdaplus))) / (2*q*zs*sigma2) 
        Gammas = sigma2 * np.abs(1 - q + q * zs * gMPs)**2 / lambdas
        xihat = xiRIE * np.maximum(1, Gammas)
        # xihat = xiRIE
        #########
        self.Creg = np.linalg.multi_dot([E.T, np.diag(xihat), E])
         
    def score(self, X):
        J = np.linalg.inv(self.Creg)
        return mm.likelihood_set_fast(J, X)

    def get_covariance(self):
        return self.Creg


class ConservativePCA(BaseEstimator):
    def __init__(self, p=10):
        self.Creg = None # come non-inizializzarla
        self.p = p
        
    def fit(self, X):
        self._fit(X)
        return self
    
    def _fit(self, Xtrain):
        T, N = np.shape(Xtrain)
        p = self.p
        Ctrain = (Xtrain.T @ Xtrain) / T

        if p == N: 
            return Ctrain

        if p >= T:
            # useful in the regime T < N 
            p = T-2

        lambdas, E = np.linalg.eigh(Ctrain)
        for i in range(len(lambdas)): 
            if np.abs(np.imag(lambdas[i])) > 1.0E-13: 
                lambdas[i] = 0.
        order = np.argsort(lambdas)[::-1]
        lambdas = lambdas[order]                     # reordering the eigvals
        E = E.T[order]                               # now each row of E is an eigvec
        
        # delta2 = np.average(lambdas[p+1:])   
        delta2 = lambdas[p]   
        diagonal = np.copy(lambdas)
        diagonal[p+1:] = delta2
        Lambda = np.diag(diagonal)
        
        Cclean = np.linalg.multi_dot([E.T, Lambda, E])
        Cdiag = 1 / np.sqrt(np.diagonal(Cclean))
        Cclean = Cdiag.reshape(-1,1) * Cclean * Cdiag.reshape(1,-1)
        #Jclean = np.linalg.inv(Cclean)
        self.Creg = Cclean
         
    def score(self, X):
        J = np.linalg.inv(self.Creg)
        return mm.likelihood_set_fast(J, X)

    def get_covariance(self):
        return self.Creg


def fit_ConservativePCA(Xtrain, Xtest, p):
    est = ConservativePCA(p).fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
    
def fit_ConservativePCA_CV(Xtrain, Xtest, cv_scoring="likelihood", n_jobs=None):
    scoring = get_scoring_function(cv_scoring)
    mycomponents = np.arange(2, Xtrain.shape[1] - 2, 1)
    cv_pca = GridSearchCV(ConservativePCA(), {'p': mycomponents}, cv=6,\
                     scoring=scoring, n_jobs=n_jobs)

    est = cv_pca.fit(Xtrain).best_estimator_

    return compute_scores(est, Xtrain, Xtest)
    

###########################################################
# Shrinkage with general biased matrix
###########################################################
###########################################################
class Shrinkage_biasedmatrix(BaseEstimator):
	def __init__(self,alpha=1.0E-3,M0=None):
		self.Creg = None # come non-inizializzarla
		self.alpha = alpha
		self.M0 = M0
        
	def fit(self, X):
		self._fit(X)
		return self
    
	def _fit(self, Xtrain):
		T, N = np.shape(Xtrain)
		alpha = self.alpha
		Ctrain = (Xtrain.T @ Xtrain) / T
		Cclean = Ctrain * (1.-alpha) + self.M0 * alpha
		self.Creg = Cclean

	def score(self, X):
		J = np.linalg.inv(self.Creg)
		return mm.likelihood_set_fast(J, X)

	def get_covariance(self):
		return self.Creg


def fit_Shrinkage_biasedmatrix(Xtrain, Xtest,alpha,M0):
    est = Shrinkage_biasedmatrix(alpha,M0).fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
    
def  fit_Shrinkage_biasedmatrix_CV(Xtrain, Xtest, M0, cv_scoring="likelihood", n_jobs=None):
    scoring = get_scoring_function(cv_scoring)
    shrinkages = np.logspace(-2, -0.1, 30)

    cv = GridSearchCV(Shrinkage_biasedmatrix(M0=M0), {'alpha': shrinkages}, cv=6,\
                     scoring=scoring, n_jobs=n_jobs) 
    est = cv.fit(Xtrain).best_estimator_
    return compute_scores(est, Xtrain, Xtest)
###########################################################
###########################################################
###########################################################


###########################################################
###########################################################
###########################################################
def fit_PCA(Xtrain, Xtest, ncomponents):
    est = PCA(svd_solver='full',n_components=ncomponents ).fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
###########################################################
###########################################################
###########################################################
    
###########################################################
###########################################################
###########################################################
def fit_FactorAnalysis(Xtrain, Xtest, ncomponents):
    est = FactorAnalysis(n_components=ncomponents, max_iter=1000).fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
    

def fit_FactorAnalysis_CV(Xtrain, Xtest, n_jobs=None, cv_scoring="likelihood"):
    scoring = get_scoring_function(cv_scoring)
    mycomponents = np.arange(2, Xtrain.shape[1], 1)
    cv = GridSearchCV(FactorAnalysis(), {'n_components': mycomponents},\
                     cv=6, scoring=scoring, n_jobs=n_jobs)
    est = cv.fit(Xtrain).best_estimator_
    return compute_scores(est, Xtrain, Xtest)
###########################################################
###########################################################
###########################################################
    

###########################################################
###########################################################
###########################################################
def fit_GraphicalLasso(Xtrain, Xtest, alpha):
    est = GraphicalLasso(max_iter=1000, alpha=alpha).fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
    
# @ignore_warnings(category=ConvergenceWarning)
def fit_GraphicalLasso_CV(Xtrain, Xtest, n_jobs=None):
    cv = GraphicalLassoCV(max_iter=1000, cv=6)
    est = cv.fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
###########################################################
###########################################################
###########################################################
    

###########################################################
###########################################################
###########################################################
def fit_Shrinkage_CV(Xtrain, Xtest, shrinkages=None, n_jobs=None, cv_scoring="likelihood"):
    scoring = get_scoring_function(cv_scoring)
    if shrinkages is None:
        shrinkages = np.logspace(-2, -0.1, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages}, cv=6,\
                     scoring=scoring, n_jobs=n_jobs) 
    est = cv.fit(Xtrain).best_estimator_
    return compute_scores(est, Xtrain, Xtest)

def fit_Shrinkage(Xtrain, Xtest, shrinkage):
    est = ShrunkCovariance(shrinkage=shrinkage).fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
###########################################################
###########################################################
###########################################################
    
###########################################################
###########################################################
###########################################################
class RIE(BaseEstimator):
    def __init__(self, eta=1,q=None,correction=True):
        self.Creg = None # come non-inizializzarla
        self.eta = eta
        self.q = q
        self.correction=correction 
    def fit(self, X):
        self._fit(X)
        return self
    
    def _fit(self,Xtrain):
        T, N = np.shape(Xtrain)
        
        C_training = np.corrcoef(Xtrain.T) 
        eta = complex(0.,1.) * self.eta / np.sqrt(N)

        lambdas, E = np.linalg.eigh(C_training)
        lambdas = np.abs(np.real(lambdas))
        order = np.argsort(lambdas)[::-1]
        lambdas = lambdas[order]                     # reordering the eigvals
        E = E.T[order]                               # now each row of E is an eigvec
        #########
        if self.q == None:
            q = N / T
        else: 
            q=self.q

        zs = lambdas - eta
        ss = np.array([np.sum(1 / (z - lambdas)) for z in zs]) / N
        ss += - 1 / (zs - lambdas) / N 
        xiRIE = lambdas / np.abs(1 - q + q*zs*ss)**2
        
        if self.correction:
            # Stieltjes transform of the Marcenko-Pastur lambdas   
            thisq=q
            sigma2 = lambdas[-1] / (1 - np.sqrt(thisq))**2
            lambdaplus = lambdas[-1] * ((np.sqrt(thisq) + 1) / (-np.sqrt(thisq) + 1))**2
            gMPs = (zs + sigma2 *(thisq-1) - np.sqrt((zs-lambdas[-1])*(zs-lambdaplus))) / (2*thisq*zs*sigma2) 

            ########################################################################################
            # Marchenko-Pastur regularisation of downwards finite-size effect for small eigenvalues:
            # c.f. to the Risk article:
            # https://www.researchgate.net/profile/Joel-Bun/publication/302339055_My_Beautiful_Laundrette_Cleaning_Correlation_Matrices_for_Portfolio_Optimization/links/572fabdf08aeb1c73d13a609/My-Beautiful-Laundrette-Cleaning-Correlation-Matrices-for-Portfolio-Optimization.pdf
            Gammas = sigma2 * np.abs(1 - q + q * zs * gMPs)**2 / lambdas
            ########################################################################################

            #print(Gammas)

            '''
            ########################################################################################
            # Inverse-Wishart regularisation of downwards finite-size effect for small eigenvalues:
            # c.f. the Physics Reports article: Physics Reports 666 (2017), p. 76
            kappa = 2*lambdas[-1]/( (1.-q-lambdas[-1])**2 -4*q*lambdas[-1] ) 
            gIWs=(zs*(1.+kappa) -kappa*(1.-q) -\ # here it is \pm \sqrt
                    ((kappa*(1.-q)-zs*(1+kappa))**2 - zs*(zs+2*q*kappa)*(2*kappa+1))**0.5 )\
                    /(zs*(zs+2.*q*kappa))

            alphaS=(1.+2.*q*kappa)**-1
            prefactor = 1.+alphaS *(lambdas-1.)
            Gammas = prefactor * np.abs(1 - q + q * zs * gIWs)**2 / lambdas
            ########################################################################################
            '''

            xihat = xiRIE * np.maximum(1, Gammas) 
        else:
            xihat = xiRIE
        #########
        self.Creg = np.linalg.multi_dot([E.T, np.diag(xihat), E])
         
    def score(self, X):
        J = np.linalg.inv(self.Creg)
        return mm.likelihood_set_fast(J, X)

    def get_covariance(self):
        return self.Creg


def fit_RIE(Xtrain, Xtest):
    est = RIE(correction=True).fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
    

def fit_RIE_CV(Xtrain, Xtest, n_jobs=None, cv_scoring="likelihood",cvq=False):
    scoring = get_scoring_function(cv_scoring)
    #est = RIE().fit(Xtrain)
    T,N=np.shape(Xtrain)
    if cvq != False:
        qs = np.arange(1.0E-2,1.,2.0E-2)
        cv = GridSearchCV(RIE(correction=True), {'q': qs}, cv=6, n_jobs=None, scoring=scoring)
    else:
        etas = np.logspace(-0.5*np.log(N)/np.log(10.),2.,num=30)
        cv = GridSearchCV(RIE(correction=True), {'eta': etas}, cv=6, n_jobs=None, scoring=scoring)
    est = cv.fit(Xtrain).best_estimator_
#    print(cv.best_params_)
    return compute_scores(est, Xtrain, Xtest)
###########################################################
###########################################################
###########################################################
    

    
###########################################################
###########################################################
###########################################################
class Oracle(BaseEstimator):
    def __init__(self, Ctrue):
        self.Ctrue = Ctrue
        
    def fit(self, X):
        return self
    
    def score(self, X):
        Creg = self.get_covariance()
        J = np.linalg.inv(Creg)
        return mm.likelihood_set_fast(J, X)

    def get_covariance(self):
        return self.Ctrue

def fit_Oracle(Ctrue, Xtrain, Xtest):
    est = Oracle(Ctrue).fit(Xtrain)
    return compute_scores(est, Xtrain, Xtest)
###########################################################
###########################################################
###########################################################



#########################################################
#########################################################
class GradientAscent(BaseEstimator):
    def __init__(self, bootstrapping=False, stop='completion', val_frac=0.1):
        self.Creg = None # come non-inizializzarla
        
        self.stop = stop
        self.bootstrapping = bootstrapping
        self.eta = 5.0E-5
        self.maxepochs = 80000
        self.training_fraction = 1 - val_frac
        self.history = {}

        
    def fit(self, X):
        self._fit(X)
        return self
    
    def _fit(self, X):
        T, N = np.shape(X)
        Xtrain = X[:int(self.training_fraction*T)]
        Xval = X[int(self.training_fraction*T):]
        Ctrain = Xtrain.T @ Xtrain / T
        Jinit = np.eye(N)
        B = int(T*0.5)

        likelihoods_tr, likelihoods_te, completions_te, self.steps, J =\
            mm.gradient_ascent_ADAM_stop(Xtrain, Ctrain, self.maxepochs, self.eta, B, Jinit,\
                                         bootstrapping=self.bootstrapping, datate=Xval, stop=self.stop)

        self.history["train_likelihood"] = likelihoods_tr
        self.history["val_likelihood"] = likelihoods_te
        self.history["val_completion_error"] = completions_te
        self.Creg = np.linalg.inv(J)        
        
    def score(self, X):
        J = np.linalg.inv(self.get_covariance())
        return mm.likelihood_set_fast(J, X)

    def get_covariance(self):
        return self.Creg


def fit_GradientAscent(Xtrain, Xtest, bootstrapping=False, stop='completion', val_frac=0.1):
    est = GradientAscent(bootstrapping=bootstrapping, stop=stop, val_frac=val_frac)
    est.fit(Xtrain)
    res = compute_scores(est, Xtrain, Xtest)
    res["history"] = est.history
    return res
#########################################################
#########################################################



#########################################################
### Gradient Ascent a' la Wishart
#########################################################
#########################################################
class GradientAscentWishart(BaseEstimator):
    def __init__(self, bootstrapping=False, stop='completion', val_frac=0.1,vectorconstraint=False):
        self.Creg = None # come non-inizializzarla
        
        self.stop = stop
        self.bootstrapping = bootstrapping
        self.eta = 1.0E-4
        self.maxepochs = 80000
        self.training_fraction = 1 - val_frac
        self.history = {}
        self.vectorconstraint=vectorconstraint

        
    def fit(self, X):
        self._fit(X)
        return self
    
    def _fit(self, X):
        T, N = np.shape(X)
        Ttrain=int(self.training_fraction*T)
        Xtrain = X[:Ttrain]
        Xval = X[Ttrain:]
        Ctrain = Xtrain.T @ Xtrain / Ttrain
        Yinit = np.eye(N)

        B = int(Ttrain*0.25)

        if self.vectorconstraint:
            likelihoods_tr_GAW,likelihoods_te_GAW,completions_te_GAW,steps_GAW,J_GAW = \
                mm.gradient_ascent_Wishart_vectormultiplier(Xtrain,Ctrain,maxepochs=self.maxepochs,eta=self.eta,B=B,Y_init=Yinit,\
                                      bootstrapping=self.bootstrapping,datate=Xval,stop=self.stop)
        
        else:
            likelihoods_tr_GAW,likelihoods_te_GAW,completions_te_GAW,steps_GAW,J_GAW = \
                mm.gradient_ascent_Wishart_multiplier(Xtrain,Ctrain,maxepochs=self.maxepochs,eta=self.eta,B=B,Y_init=Yinit,\
                                      bootstrapping=self.bootstrapping,datate=Xval,stop=self.stop)
        

        self.history["train_likelihood"] = likelihoods_tr_GAW
        self.history["val_likelihood"] = likelihoods_te_GAW
        self.history["val_completion_error"] = completions_te_GAW
        self.Creg = np.linalg.inv(J_GAW)        
        
    def score(self, X):
        J = np.linalg.inv(self.get_covariance())
        return mm.likelihood_set_fast(J, X)

    def get_covariance(self):
        return self.Creg


def fit_GradientAscentWishart(Xtrain, Xtest, bootstrapping=False, stop='completion', val_frac=0.1):
    est = GradientAscentWishart(bootstrapping=bootstrapping, stop=stop, val_frac=val_frac)
    est.fit(Xtrain)
    res = compute_scores(est, Xtrain, Xtest)
    res["history"] = est.history
    return res
#########################################################
### END Gradient Ascent a' la Wishart
#########################################################
#########################################################
