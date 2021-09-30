import numpy as np
import random


def likelihood_samples(J,samples):
    N=len(J)
    T=len(samples)
    likelihoods=np.zeros(T)
    likelihoods-=0.5*np.sum(samples*np.dot(samples,J),axis=1)
    likelihoods+=0.5*np.sum(np.log(np.real(np.linalg.eigvals(J))))
    likelihoods-=0.5*N*np.log(2*np.pi)
    return likelihoods

def likelihood_set(J,data):
    return np.average(likelihood_samples(J,data))

def likelihood_set_fast(J,data):
    medie=np.average(data,axis=0)

    T,N=np.shape(data)
    C=np.dot(data.T,data)/T
    somma=np.trace(np.dot(C,J))
#    somma+=np.linalg.multi_dot([medie.T,J,medie]) 
#    somma+=np.dot(medie.T , np.dot(J,medie) ) 
#    somma+=np.sum(medie*np.dot(medie,J)  )
    auto=np.real(np.linalg.eigvals(J))
    return -0.5*(somma+N*np.log(2*np.pi)-np.sum(np.log(auto)))

def energy_set(J,C):
    R=len(J)
    somma=np.trace(np.dot(C,J))
    return -0.5*(somma)

'''
#this is not the reconstruction, but the "completion" error
def reconstruction_meansquareerror(J,dati):
    T,N=np.shape(dati)
    somma=0.
    precisioni=np.diag(J)
    matrice=np.eye(N)+(J-np.diag(precisioni))/precisioni
    for dato in dati:
        somma+=(np.linalg.norm(np.dot(matrice,dato)))/np.linalg.norm(dato)
    return somma/(np.sqrt(N)*T)
'''

#this is not the reconstruction, but the "completion" error
def reconstruction_meansquareerror(J,dati):
    T,N=np.shape(dati)
    precisioni=np.diag(J)
    matrice=np.eye(N)+(J-np.diag(precisioni))/precisioni
    myvector= [np.average(np.abs(np.dot(matrice,vector))) for vector in dati]
    return np.average(myvector)



def KLdivergence(C1,C2):
    D=len(C1)
 #   detC1=np.linalg.det(C1)
    detC2=np.linalg.det(C2)

    w1=np.linalg.eigvals(C1)
    lambdas1=w1[np.where(w1>1.0E-14)[0]]
    logdetC1=np.sum(np.log(lambdas1))

    C2inv=np.linalg.inv(C2)
 #   aux=np.dot( np.dot(E1.T , C2inv ) , E1 ) 
 #   aux=np.diag(aux)
    #term2=np.dot(w1,aux)
    #term2=(1.*D - term2)

    term2=1.*D - np.trace( np.dot( C2inv , C1 ) ) 
    term1=logdetC1-np.log(detC2)

    return -0.5*(+term1+term2)/D



from scipy.stats import ortho_group

def randompositivedefinitematrix1(dim,rank,alpha=1.,diagonal=False,constantspectrum=False):
    if constantspectrum:
        lambdas=np.ones(dim)*(1.*dim/rank)
        epsilons=np.ones(dim)*(1.*rank/dim)
        lambdas[rank:]=0.
        epsilons[rank:]=0.
    else:
        aux=np.random.dirichlet(alpha*np.ones(rank))
        lambdas =np.concatenate( ( dim*aux       , np.zeros(dim-rank) )  )
        epsilons=np.concatenate( ( (dim*aux)**-1 , np.zeros(dim-rank) )  )
        
    indices =np.argsort(lambdas)
    lambdas =lambdas[indices]
    epsilons=epsilons[indices]

    if diagonal:
        E=np.eye(dim)
    else:
        E=ortho_group.rvs(dim)
        
    C=np.dot(E.T, np.dot(np.diag(lambdas ),E) )
    J=np.dot(E.T, np.dot(np.diag(epsilons),E) )
#    plt.plot(np.sort(dim*lambdas),'s-')
    return C,J,E,lambdas

def standardize_matrix(C):
    aux=np.outer(np.diag(C),np.diag(C)) 
    return C/(aux)**(0.5)

def generateNullDataset(dim:int, rank:int, S:int, alpha=1., diagonal=False, generateJ=False):
    
    if generateJ:
        J,C,E,epsilons = randompositivedefinitematrix1(dim,rank,alpha,diagonal=diagonal)
        lambdas=epsilons**-1.
    else:
        C,J,E,lambdas = randompositivedefinitematrix1(dim,rank,alpha,diagonal=diagonal)

    mu = np.zeros(dim)
    s = 1 / np.sqrt(np.diag(C))
    C = s.reshape(-1,1) * C * s.reshape(1,-1)
    newxs = np.random.multivariate_normal(mu, C, S)  # shape S x dim
    return newxs,[C,E,lambdas]


#one realization of C with bootstrapping
def corr2_bootstrap(data,N):
    T=np.shape(data)[0]
    indiciv=T*np.random.sample((N)) #creo un campione di N numeri random compresi fra 0 e T 
    indiciv=np.array(indiciv,dtype=int) #metto il campione in un array prendendone la parte intera
    dataB=data[indiciv]
    dataB-=np.average(dataB,axis=0)
    dataB/=np.std(dataB,axis=0,ddof=1)
    return np.corrcoef(dataB.T)

#many realizations of C with bootstrapping
def averageC(data,N,nbJs):
    D=np.shape(data)[1]
    avC=np.zeros((D,D))
    for s in range(nbJs):
        myC=corr2_bootstrap(data,N)
        avC+=myC
    return avC/nbJs


################################
## functions for mini-batch learning 
################################
################################
def gradient_ascent(C_training,C_test,epochs,eta,B,J_init,bootstrapping=False):
    J_epoch=np.copy(J_init)
    likelihoods_tr=[]
    likelihoods_te=[]
    steps=[]
    
    tsamp=epochs/100  
    
    if bootstrapping:
        for epoch in range(epochs):
            
            if epoch%tsamp==0:
                likelihoods_tr.append( m.likelihood_set(J_epoch,C_training) )
                likelihoods_te.append( m.likelihood_set(J_epoch,C_test) )
                steps.append(epoch)

            J_epoch-=eta*( corr2_bootstrap(data,B) - np.linalg.inv(J_epoch) )

    else:
        for epoch in range(epochs):
            
            if epoch%tsamp==0:
                likelihoods_tr.append( m.likelihood_set(J_epoch,C_training) )
                likelihoods_te.append( m.likelihood_set(J_epoch,C_test) )
                steps.append(epoch)

            J_epoch-=eta*( C_training - np.linalg.inv(J_epoch))

    
    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),J_epoch

def gradient_ascent_Lasso(C_training,C_test,epochs,eta,B,J_init,lambda_lasso=1.0E-2,bootstrapping=False):
    J_epoch=np.copy(J_init)
    likelihoods_tr=[]
    likelihoods_te=[]
    steps=[]
    
    tsamp=epochs/100  
    
    if bootstrapping:
        for epoch in range(epochs):
            
            if epoch%tsamp==0:
                likelihoods_tr.append( m.likelihood_set(J_epoch,C_training) )
                likelihoods_te.append( m.likelihood_set(J_epoch,C_test) )
                steps.append(epoch)

            signsmatrix=np.sign(J_epoch)
            J_epoch-=eta*( corr2_bootstrap(data,B) - np.linalg.inv(J_epoch) + lambda_lasso*signsmatrix )

    else:
        for epoch in range(epochs):
            
            if epoch%tsamp==0:
                likelihoods_tr.append( m.likelihood_set(J_epoch,C_training) )
                likelihoods_te.append( m.likelihood_set(J_epoch,C_test) )
                steps.append(epoch)
                
            signsmatrix=np.sign(J_epoch)
            J_epoch-=eta*( C_training - np.linalg.inv(J_epoch) + lambda_lasso*signsmatrix ) 

    
    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),J_epoch

#with an adaptative correction of the learning rate (not using it)
def gradient_ascent_test(C_training,C_test,epochs,eta,B,J_init,bootstrapping=False,t_eta=100):
    J_epoch=np.copy(J_init)
    likelihoods_tr=[]
    likelihoods_te=[]
    steps=[]
    
    min_evals=[]
    etas=[]
    
    tsamp=epochs/100  
    
    if bootstrapping:
        for epoch in range(epochs):

            deltaJ=-eta*( corr2_bootstrap(data,B) - np.linalg.inv(J_epoch))
            min_eval=LA.eigsh(J_epoch+deltaJ,k=1,return_eigenvectors=False, which='SA')[0]
            if min_eval>0.:
                J_epoch+=deltaJ 
            else: eta*=0.9

            if epoch%t_eta==0:
                eta*=1.1
                        
            if epoch%tsamp==0:
                likelihoods_tr.append( m.likelihood_set(J_epoch,C_training) )
                likelihoods_te.append( m.likelihood_set(J_epoch,C_test) )
                steps.append(epoch)
                min_evals.append( min_eval )
                etas.append(eta)
                #min_eval.append( np.min(np.linalg.eig(J_epoch)[0]) )
       
    else:
        for epoch in range(epochs):

            
            deltaJ=-eta*( corr2_bootstrap(data,B) - np.linalg.inv(J_epoch))
            min_eval=LA.eigsh(J_epoch+deltaJ,k=1,return_eigenvectors=False, which='SA')[0]
                #min_eval.append( np.min(np.linalg.eig(J_epoch)[0]) )
            if min_eval>0.:
                J_epoch+=deltaJ
            else: eta*=0.1

            if epoch%t_eta==0:
                eta*=1.1

            if epoch%tsamp==0:
                likelihoods_tr.append( m.likelihood_set(J_epoch,C_training) )
                likelihoods_te.append( m.likelihood_set(J_epoch,C_test) )
                steps.append(epoch)
                min_evals.append( min_eval )
                etas.append(eta)



    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),J_epoch,np.array(min_evals),np.array(etas)


#c.f. arxiv.org/abs/1212.0901v2
def gradient_ascent_Nesterov(C_training,C_test,epochs,eta,B,J_init,bootstrapping=False,increasing_eta=False):
    J_epoch=np.copy(J_init)
    D=len(J_init)
    v=np.zeros((D,D))
    epsilon=eta
    mu=2.5E-1
    mu2=mu**2
    oneplusmuepsilon=(1.+mu)*epsilon
    
    likelihoods_tr=[]
    likelihoods_te=[]
    steps=[]
    
    tsamp=epochs/100  
    
    eta0=eta
    
    if bootstrapping:
        for epoch in range(epochs):
            
            if increasing_eta:
            #epsilon=eta0*np.exp(-1.*epoch*np.log(3.)/epochs)
                epsilon=eta0+eta0*epoch/(1.*epochs)
                oneplusmuepsilon=(1.+mu)*epsilon

            if epoch%tsamp==0:
                likelihoods_tr.append( m.likelihood_set(J_epoch,C_training) )
                likelihoods_te.append( m.likelihood_set(J_epoch,C_test) )
                steps.append(epoch)

            gradient=corr2_bootstrap(data,B) - np.linalg.inv(J_epoch)
            J_epoch = J_epoch +mu2*v - oneplusmuepsilon*gradient
            v = mu*v - eta*gradient 


    else:
        for epoch in range(epochs):
            if increasing_eta:
            #epsilon=eta0*np.exp(-1.*epoch*np.log(3.)/epochs)
                epsilon=eta0+eta0*epoch/(1.*epochs)
                oneplusmuepsilon=(1.+mu)*epsilon

            if epoch%tsamp==0:
                likelihoods_tr.append( m.likelihood_set(J_epoch,C_training) )
                likelihoods_te.append( m.likelihood_set(J_epoch,C_test) )
                steps.append(epoch)

            gradient=C_training - np.linalg.inv(J_epoch)
            J_epoch = J_epoch +mu2*v - oneplusmuepsilon*gradient
            v = mu*v - eta*gradient 




    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),J_epoch

#c.f. arxiv.org/pdf/1412.6980.pdf
def gradient_ascent_ADAM(data,C_training,C_test,epochs,eta,B,J_init,bootstrapping=False):
    J_epoch=np.copy(J_init)
    D=len(J_init)
    v=np.zeros((D,D))
    mom=np.zeros((D,D))
    
    epsilon=1.0E-6
    beta1=0.9
    beta2=0.999
    
    likelihoods_tr=[]
    likelihoods_te=[]
    steps=[]
        
    tsamp=epochs/100  
    
    eta0=eta
    
    if bootstrapping:
        for epoch in range(epochs):
    
            if epoch%tsamp==0:
                likelihoods_tr.append( likelihood_set(J_epoch,C_training) )
                if C_test is not False:
                    likelihoods_te.append( likelihood_set(J_epoch,C_test) )
                steps.append(epoch)
                
            gradient=corr2_bootstrap(data,B) - np.linalg.inv(J_epoch)
            mom = beta1*mom + (1.-beta1)*gradient
            v = beta2*v + (1.-beta2)*gradient**2
            mhat = mom/(1.-beta1**(epoch+1.))
            vhat = v/(1.-beta2**(epoch+1.))

            J_epoch = J_epoch - eta*mhat/(np.sqrt(vhat)+epsilon)


    else:
        for epoch in range(epochs):

            if epoch%tsamp==0:
                likelihoods_tr.append( likelihood_set(J_epoch,C_training) )
                if C_test is not False:
                    likelihoods_te.append( likelihood_set(J_epoch,C_test) )
                steps.append(epoch)
                
            gradient=C_training - np.linalg.inv(J_epoch)
            mom = beta1*mom + (1.-beta1)*gradient
            v = beta2*v + (1.-beta2)*gradient**2
            mhat = mom/(1.-beta1**(epoch+1.))
            vhat = v/(1.-beta2**(epoch+1.))

            J_epoch = J_epoch - eta*mhat/(np.sqrt(vhat)+epsilon)

    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),J_epoch


#in diagonal approximation
def logdetFisher(C):
    N=len(C)
    somma=0.
    for i in range(N):
        for j in range(i+1):
            somma+=np.log(C[i,i]*C[j,j]+C[i,j]*C[i,j])
    return somma

#ONLY THE HESSIAN ON -<H>. S=EMPIRICAL CORRELATION
def logdetFisher_wrtC(J,S):
    N=len(C)
    somma=0.
    for i in range(N):
        for j in range(i+1):
            somma+=np.log(C[i,i]*C[j,j]+C[i,j]*C[i,j])
    return somma


def completion_vector_inC(myC,myx):
    myJ=np.linalg.inv(myC)
    precisioni=np.diag(myJ)
    matrice=-(myJ-np.diag(precisioni))/precisioni
    return np.dot(matrice,myx)

def completion_vector_inJ(myJ,myx):
    precisioni=np.diag(myJ)
    matrice=-(myJ-np.diag(precisioni))/precisioni
    return np.dot(matrice,myx)




#c.f. arxiv.org/pdf/1412.6980.pdf
# 'stop' must be ['completion','likelihood','False']
# the condition 'datate==False' is equivalent to 'stop==False'
def gradient_ascent_ADAM_stop(data,C_training,maxepochs,eta,B,J_init,bootstrapping=False,T=0.,datate=False,stop=False):
    J_epoch=np.copy(J_init)
    D=len(J_init)
    v=np.zeros((D,D))
    mom=np.zeros((D,D))
    
    epsilon=1.0E-6
    beta1=0.9
    beta2=0.999
    
    likelihoods_tr=[]
    likelihoods_te=[]
    completions_te=[]
    steps=[]
        
    if maxepochs>100:
        tsamp=maxepochs/100
    else:
        tsamp=10
    
    eta0=eta
    
    decreasing=False
    epoch=1

    behind_steps=4
    J_past=np.zeros((behind_steps,D,D))
    
    while epoch < maxepochs and not decreasing:
        epoch+=1

        if bootstrapping:
            gradient=corr2_bootstrap(data,B) - np.linalg.inv(J_epoch)
        else:
            gradient=C_training - (1.+T)*np.linalg.inv(J_epoch)

        mom = beta1*mom + (1.-beta1)*gradient
        v = beta2*v + (1.-beta2)*gradient**2
        mhat = mom/(1.-beta1**(epoch+1.))
        vhat = v/(1.-beta2**(epoch+1.))

        J_epoch = J_epoch - eta*mhat/(np.sqrt(vhat)+epsilon)


        if epoch%tsamp==0:
            likelihoods_tr.append( likelihood_set(J_epoch,C_training) )

            if datate is not False:
                myvector=[np.average(np.abs(vector-completion_vector_inJ(J_epoch,vector))) for vector in datate]
                thiscompletion=np.average(myvector)
                completions_te.append(thiscompletion)

                likelihoods_te.append( likelihood_set(J_epoch,datate) )

                if stop=='completion':
                    if len(likelihoods_te) > behind_steps:
                        logic_array=[ completions_te[-1]>pippo for pippo in completions_te[-behind_steps:-1] ]
                        if not False in logic_array: decreasing=True
                    if stop=='likelihood':
                        logic_array=[ likelihoods_te[-1]<pippo for pippo in likelihoods_te[-behind_steps:-1] ]
                        if not False in logic_array: decreasing=True

            steps.append(epoch)

            for i in range(behind_steps-1):
                J_past[behind_steps-i-1]=np.copy(J_past[behind_steps-i-2])
            J_past[0]=np.copy(J_epoch)


 #    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),J_epoch
    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(completions_te),np.array(steps),J_past[behind_steps-2]





##########################################
##########################################
# gradient ascent in C
##########################################
#c.f. arxiv.org/pdf/1412.6980.pdf
def gradient_ascent_inC_ADAM_stop(data,C_training,C_test,maxepochs,eta,B,C_init,bootstrapping=False,return_Fisher=False,T=0.,stop=False):
    C_epoch=np.copy(C_init)
    J_epoch=np.linalg.inv(C_epoch)
    N=len(C_init)
    v=np.zeros((N,N))
    mom=np.zeros((N,N))
    
    epsilon=1.0E-6
    beta1=0.9
    beta2=0.999
    
    likelihoods_tr=[]
    likelihoods_te=[]
    steps=[]
        
    if return_Fisher:
        log_det_Fisher=[]

    tsamp=maxepochs/100
    eta0=eta
    decreasing=False
    epoch=1
    behind_steps=4
    C_past=np.zeros((behind_steps,N,N))
    
    if bootstrapping:
        while epoch < maxepochs and not decreasing:
            epoch+=1

            Sigma_empirical=corr2_bootstrap(data,B) # potrei creare una variante di questa funzione che usa i T-B scartati come testset (non validation)
            gradient=np.linalg.multi_dot([J_epoch, Sigma_empirical - C_epoch , J_epoch]) 

            mom = beta1*mom + (1.-beta1)*gradient
            v = beta2*v + (1.-beta2)*gradient**2
            mhat = mom/(1.-beta1**(epoch+1.))
            vhat = v/(1.-beta2**(epoch+1.))

            C_epoch += eta*mhat/(np.sqrt(vhat)+epsilon)
##################
#            aux=np.outer(np.diag(C_epoch),np.diag(C_epoch)) 
#            C_epoch/=aux**(0.5)   # matrix standardisation
##################

            C_epoch *= N/np.trace(C_epoch)
            J_epoch=np.linalg.inv(C_epoch)
   
            if epoch%tsamp==0:
                likelihoods_tr.append( likelihood_set(J_epoch,C_training) )
                if C_test is not False:
                    likelihoods_te.append( likelihood_set(J_epoch,C_test) )

                    if stop:
                        if len(likelihoods_te) > behind_steps:
                            logic_array=[ likelihoods_te[-1]<pippo for pippo in likelihoods_te[-behind_steps:-1] ]
                            if not False in logic_array: decreasing=True
                        #if likelihoods_te[-1]<likelihoods_te[-4] and likelihoods_te[-2]<likelihoods_te[-4] and likelihoods_te[-3]<likelihoods_te[-4]: decreasing=True
                
                steps.append(epoch)

                for i in range(behind_steps-1):
                    C_past[behind_steps-i-1]=np.copy(C_past[behind_steps-i-2])
                C_past[0]=np.copy(C_epoch)

    else:
        while epoch < maxepochs and not decreasing:
            epoch+=1

            gradient=np.linalg.multi_dot([J_epoch, C_training - C_epoch , J_epoch]) 

            mom = beta1*mom + (1.-beta1)*gradient
            v = beta2*v + (1.-beta2)*gradient**2
            mhat = mom/(1.-beta1**(epoch+1.))
            vhat = v/(1.-beta2**(epoch+1.))

            C_epoch += eta*mhat/(np.sqrt(vhat)+epsilon)
            C_epoch *= N/np.trace(C_epoch)
            J_epoch=np.linalg.inv(C_epoch)

            
            if epoch%tsamp==0:
                likelihoods_tr.append( likelihood_set(J_epoch,C_training) )
                if C_test is not False:
                    likelihoods_te.append( likelihood_set(J_epoch,C_test) )
                
                    if stop:
                        if len(likelihoods_te) > behind_steps:
                            logic_array=[ likelihoods_te[-1]<pippo for pippo in likelihoods_te[-behind_steps:-1] ]
                            if not False in logic_array: decreasing=True
                    #    if likelihoods_te[-1]<likelihoods_te[-4] and likelihoods_te[-2]<likelihoods_te[-4] and likelihoods_te[-3]<likelihoods_te[-4]: decreasing=True

                steps.append(epoch)
                
                for i in range(behind_steps-1):
                    C_past[behind_steps-i-1]=np.copy(C_past[behind_steps-i-2])
                C_past[0]=np.copy(C_epoch)

                if return_Fisher:
                    log_det_Fisher.append(logdetFisher(C_epoch))

 #    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),J_epoch

    if return_Fisher:
        return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),C_past[behind_steps-2],log_det_Fisher
    return np.array(likelihoods_tr),np.array(likelihoods_te),np.array(steps),C_past[behind_steps-2]
##########################################
##########################################
##########################################

################################
## end functions for mini-batch learning 
################################
################################




def BootstrapErrorC2_naive(data,B=100):
    data=np.copy(data)
    [T,R]=np.shape(data)
    Clist=[]
    
    for b in range(B):
        
        indiciv=T*np.random.sample((T)) #creo un campione di T numeri random compresi fra 0 e T (170 passi temporali)
        indiciv=np.array(indiciv,dtype=int) #metto il campione in un array prendendone la parte intera

        datib=data[indiciv]
        
        thisC=np.corrcoef(datib.T)
        Clist.append(thisC) 
            
    Clist=np.array(Clist)
    
    Cav=np.average(Clist,axis=0)
    Cstd=np.std(Clist,axis=0)
        
    return Cav ,Cstd


def matcorrlambdapiatto(C,spettro=True): #lo presi da FrancescaCodes/moduli.py 
    w1,v1=np.linalg.eig(C)
    lamb=np.real(np.amin(w1)) #lambda e' il modulo del piu' piccolo autoval di C (quindi il piu' grande autoval negativo in modulo)
    
    if lamb<0:
        indici=np.where(w1<=abs(lamb))[0]
        w1[indici]=abs(lamb)
       
        D=len(C)
        aux=np.dot(v1,np.eye(D)*w1)  
        Mc=np.dot(aux,v1.T) 
    else:
        Mc=np.copy(C)
    
    aux=np.outer(np.diag(Mc),np.diag(Mc))
    Mc=Mc/(aux)**(0.5)
    if spettro:
        return Mc,w1
    else:
        return Mc


#########################################################
### regularisation strategies
#########################################################
#########################################################
#########################################################
#########################################################
def regularisation_average(trainingset,C_training,C_test):
    D=len(C_training)
    batchsizes = np.arange(4,D/2,1)

    likelihoods_tr_average=[]
    likelihoods_te_average=[]
    #batchsizes = [ D/n for n in range(10,2,-1) ]
    for BS in batchsizes:
        avC=averageC(trainingset,BS,1000)
        J_average=np.linalg.inv(avC)
        likelihoods_tr_average.append( m.likelihood_set(J_average,C_training) )
        likelihoods_te_average.append( m.likelihood_set(J_average,C_test) )

    max_average_te=np.max(likelihoods_te_average[:])
    indice_average=np.argmax(likelihoods_te_average[:])
    max_average_tr=likelihoods_tr_average[indice_average]
    batch_max=batchsizes[indice_average]
    avC=averageC(trainingset,batch_max,1000)
    bestJ_average=np.linalg.inv(avC)

    return max_average_te,max_average_tr,batch_max,bestJ_average

def regularisation_diagonal(C_training,C_test):
    Lambda_max=1.
    Lambda_min=5.0E-3
    delta=(Lambda_max-Lambda_min)/200
    Lambdas=np.arange(Lambda_min,Lambda_max,delta)
    likelihoods_tr_diagonal=[]
    likelihoods_te_diagonal=[]
    for Lambda in Lambdas:
        myC=C_training+Lambda*np.eye(len(C_training)) 
        aux=np.outer(np.diag(myC),np.diag(myC)) 
        myC=myC/(aux)**(0.5)   # matrix standardisation
        myJ=np.linalg.inv(myC)
        likelihoods_tr_diagonal.append(m.likelihood_set(myJ,C_training))
        likelihoods_te_diagonal.append(m.likelihood_set(myJ,C_test))

    max_diagonal_te=np.max(likelihoods_te_diagonal[:])
    indice_diagonal=np.argmax(likelihoods_te_diagonal[:])
    max_diagonal_tr=likelihoods_tr_diagonal[indice_diagonal]
    Lambda_max=Lambdas[indice_diagonal]

    ###
    myC=C_training+Lambda_max*np.eye(len(C_training)) 
    aux=np.outer(np.diag(myC),np.diag(myC)) 
    myC=myC/(aux)**(0.5)   # matrix standardisation
    bestJ_diagonal=np.linalg.inv(myC)
    
    return max_diagonal_te,max_diagonal_tr,Lambda_max,bestJ_diagonal

def regularisation_RSSC(trainingset,C_training,C_test):
    delta_min=5.0E-3
    delta_max=1.
    delta_delta=(delta_max-delta_min)/200
    deltas=np.arange(delta_min,delta_max,delta_delta)
    inference_training=inference_pairwise_significativitysoil(trainingset)
    myJs=inference_training.computeJs(deltas)
    likelihoods_tr_RSSC=np.array([m.likelihood_set(thisJ,C_training) for thisJ in myJs])
    likelihoods_te_RSSC=np.array([m.likelihood_set(thisJ,C_test) for thisJ in myJs])
    
    max_RSSC_te=np.max(likelihoods_te_RSSC[:])
    indice_RSSC=np.argmax(likelihoods_te_RSSC[:])
    max_RSSC_tr=likelihoods_tr_RSSC[indice_RSSC]
    delta_max=deltas[indice_RSSC]
    bestJ_RSSC=myJs[indice_RSSC]

    return max_RSSC_te,max_RSSC_tr,delta_max,bestJ_RSSC

#########################################################
#########################################################
#########################################################
#########################################################




class  inference_pairwise_significativitysoil():
    def __init__(self,data):
        #data correlation function Bootstrap error
        ##############################################
        Cav,Cerr=BootstrapErrorC2_naive(data,B=100)
        self.C_P=np.corrcoef(data.T)
        self.R=len(Cerr)
        self.T_val=np.abs(self.C_P)/(Cerr + np.eye(self.R)) #there was an error here in Francesca's codes
        ##############################################


    def computeJ(self,delta,returnJ=True):
        indici=np.where( self.T_val < delta )
        C_delta=np.copy(self.C_P-np.eye(self.R))
        for el in zip(indici[0],indici[1]):    
            C_delta[el]=0. 
        C_s=C_delta+np.eye(self.R)                              #eliminare la somma e la sottrazione della diagonale
        mat,spectrum= matcorrlambdapiatto(C_s, spettro=True)  #questa funzione va vista
        if returnJ:
            J_s=np.linalg.inv(mat) 
            return J_s,C_s,spectrum
        else:
            return mat

    def computeJs(self,deltas):
#        return map(self.computeJ,deltas)
        return np.array([self.computeJ(delta)[0] for delta in deltas])


def naif_matrix_distance(A,B):
    return np.sqrt(np.sum( (A-B)**2 ))/(1.*len(A))
