import numpy as np
from os.path import isfile
import moduli_miguel as mm

def get_dati_tommaso(standardize=True):
    path='dati_Tommaso_sani/'
    data_subjects = {}
    for i in range(1,65):
        nomefile = path+'subj'+str(i)+'.rs_timesries_HC.txt'
        if isfile(nomefile):
            this_data = np.loadtxt(nomefile)
            data_subjects[i] = this_data
    
    data_subjects_zeromean = {}
    for i, sub in data_subjects.items():
        data_subjects_zeromean[i] = (sub - np.mean(sub,axis=0)) / np.std(sub,axis=0)
    
    if standardize:
        return data_subjects_zeromean
    else:
        return data_subjects

def generate_data_dirichlet(*, Ns=40, T=180, N=116, alpha=1.):
    
    #training_fraction= ... # 
    #(It must coincide with that of the real data)
    #(so I don't initialize it)
    Xall, Call, Uall, lambdas = {}, {}, {}, {}
    for i in range(Ns):
        Xall[i],[Call[i], Uall[i], lambdas[i]] = mm.generateNullDataset\
                    (N,N,T,alpha=alpha, diagonal=False,generateJ=True)
    return Xall, Call, Uall, lambdas


def split_train_test(X, train_fraction, standardize=False, shuffle=True):
    X = X.copy()
    if shuffle:
        np.random.shuffle(X)
    T, N = X.shape
    Xtrain = X[:int(train_fraction*T)]
    Xtest = X[int(train_fraction*T):]
    if standardize:
        avg = np.mean(Xtrain, axis=0)
        Xtrain -= avg 
        std = np.std(Xtrain, axis=0)
        Xtrain /= std
        
        Xtest -= avg
        Xtest /= std


    return Xtrain, Xtest

def computeCbar(database,train_fraction=0.8):
    #indextoexclude=[21,23,28,31,32]
    indextoexclude=[]
    Ns,T,N=np.shape(database)
    Cs=np.zeros((Ns-len(indextoexclude),N,N))

    i=0
    for subi,data_subject in enumerate(database):
        if subi not in indextoexclude:
            Xtrain, Xtest=split_train_test(data_subject,train_fraction=train_fraction,standardize=True,shuffle=True)
            Cs[i]=np.corrcoef(Xtrain.T)

            i+=1
    Cbar=np.average(Cs,axis=0)
    return Cbar

def completion_tvalue(dataset,myC):
    myJ=np.linalg.inv(myC)
    precisioni=np.diag(myJ)
    matrice=-(myJ-np.diag(precisioni))/precisioni
    myvector = [np.average(np.sqrt(precisioni)*np.abs(vector-np.dot(matrice,vector))) for vector in dataset]
    return np.average(myvector),np.std(myvector,ddof=1)/np.sqrt(len(myvector))


def merge_dicts(a, b, path=None, keepa=False, keepb=False):
    """
    Merge two dictionaries recursively
    modifying the content of `a`.
    If keepa=True keep the content of a in case of conflicting keys
    If keepb=True keep the content of b in case of conflicting keys
    """  
    if path is None: 
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)], keepa, keepb)
            elif keepa and not keepb:
                pass
            elif not keepa and keepb:
                a[key] = b[key]
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def splat_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for v2 in splat_dict(v):
                yield (k, *v2)
        else:
            yield (k, v)
