%matplotlib inline
#This will ensure any plt matplotlib calls automatically output correctly.
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import sys
import sklearn
import pandas
from pandas.tools.plotting import scatter_matrix
from sklearn import preprocessing

def logisreg4(param0, datax, datay):
#This is the function to obtain the parameters for our regression model.
    LB = 10**(-5)
    UB = 1 - LB
    Nones = [(None,None)] * (len(param0)-2)
    ourbounds = [(0,1),(0,1)]
    ourbounds.extend(Nones)
    result = scipy.optimize.minimize(logis4regMLE,param0,args=(datax,datay),method='L-BFGS-B',bounds=ourbounds)
    #result = scipy.optimize.minimize(logis4regMLE,param0,args=(datax,datay),method='Nelder-Mead')
    return result.x,result.nit;

def logis4regMLE(parameters,datax,datay):
#This is the 4 Parameter logistic regression likelihood function without a for loop.
#Parameters should be an array with the following composition [c, b1, LA, RA]
#The larger the output the worse the likelihood.
    loglikelihood = 0
    LB = 10**(-5)
    UB = 1 - LB
    LA = parameters[0]
    RA = parameters[1]
    LA = max(LB, min(UB,LA))
    RA = min(UB, max(LB,RA))
    score=logis4score(parameters,datax)
    predictedprob = LA+((RA-LA)/(1+np.exp(score)))
    loglikelihood = np.sum( datay * np.log(predictedprob) + (1-datay)*np.log(1-predictedprob))
    return -loglikelihood;

def logis4score(parameters,datax):
    b = np.zeros(len(parameters)-2)
    for i in range(2,len(parameters)):
        b[i-2] = parameters[i]
    score=-np.dot(datax,b)
    for i in range(0,len(score)):
        if score[i]>30:
            score[i]=30
        if score[i]<-30:
            score[i]=-30
    return score;

def logis2regMLE(parameters,datax,datay):
#This is the 2 Parameter logistic regression log-likelihood function without a for loop.
#Parameters should be an array with the following composition [c, b1]
#The larger the output the worse the likelihood.
    loglikelihood = 0
    score=logis2score(parameters,datax)
    predictedprob = 1/(1+np.exp(score))
    loglikelihood = np.sum( datay * np.log(predictedprob) + (1-datay)*np.log(1-predictedprob))
    return -loglikelihood;


def logisreg2(param0,datax, datay):
#This is the function to obtain the parameters for our two parameter regression model.
    Nones = [(None,None)] * (len(param0))
    result = scipy.optimize.minimize(logis2regMLE,param0,args=(datax,datay),method='L-BFGS-B',bounds=Nones)
    #result = scipy.optimize.minimize(logis2regMLE,param0,args=(datax,datay),method='Nelder-Mead')
    return result.x,result.nit;

def logis2score(parameters,datax):
    b = np.zeros(len(parameters))
    for i in range(0,len(parameters)):
        b[i] = parameters[i]
    score=-np.dot(datax,b)
    for i in range(0,len(score)):
        if score[i]>30:
            score[i]=30
        if score[i]<-30:
            score[i]=-30
    return score;

def crossvalidate(reps,datax,datay,parameters2,parameters4):
    #This function will do the cross validation for us.
    archive2res = np.zeros((reps,len(parameters2)))
    archive2it = np.zeros((reps,1))
    archive2ll = np.zeros(reps)
    archive2score = np.zeros((reps,len(datax)))
    archive4res = np.zeros((reps,len(parameters4)))
    archive4it = np.zeros((reps,1))
    archive4ll = np.zeros(reps)
    archive4score = np.zeros((reps,len(datax)))
    for i in range(0,reps):
        np.random.seed(seed=2017+i)
        train = np.random.rand(len(datay)) <= .5
        test = 1 - train
        result2 = logisreg2(parameters2,datax[train],datay[train])
        ll2test = logis2regMLE(result2[0],datax[test],datay[test])
        score2 = logis2score(result2[0],datax[test])
        result4 = logisreg4(parameters4,datax[train],datay[train])
        ll4test = logis4regMLE(result4[0],datax[test],datay[test])
        score4 = logis4score(result4[0],datax[test])
        archive2res[i,]=result2[0]
        archive2it[i,]=result2[1]
        archive2ll[i]=ll2test
        archive2score[i,]=score2
        archive4res[i,]=result4[0]
        archive4it[i,]=result4[1]
        archive4ll[i]=ll4test
        archive4score[i,]=score4
    return archive2res,archive2ll,archive2score,archive4res,archive4ll,archive4score;

def archivescatter(archive2res,archive2ll,archive2pp,archive4res,archive4ll,archive4pp):
    diffll=archive2ll-archive4ll
    plt.scatter(x=range(0,len(diffll)),y=diffll)
    plt.show()
    sum(diffll)
    plt.scatter(x=archive2res[:,0],y=archive2res[:,1])
    plt.show()
    plt.scatter(x=archive4res[:,0],y=archive4res[:,1])
    plt.show()
    fig = plt.figure()
    combscat = fig.add_subplot(111)
    combscat.scatter(x=archive2res[:,0],y=archive2res[:,1], s=10, c='b', marker="s", label='2 Parameters')
    combscat.scatter(x=archive4res[:,0],y=archive4res[:,1], s=10, c='r', marker="o", label='4 Parameters')
    plt.legend(loc='upper right')
    plt.show()
    return;

def seperatedata(matrixin,numattributes,observations,yloc):
    xvalues = np.ones((observations,numattributes))
    yvalues = np.zeros(observations)
    for j in range(0,observations):
        for i in range(0,numattributes):
                if i == yloc:
                    yvalues[j]=matrixin[j,i]
                else:
                    xvalues[j,i+1] = matrixin[j,i]
    return xvalues,yvalues;

def grabdata(filename,numattributes,observations,yloc):
    data=open(filename,'r')
    xvalues = np.ones((observations,numattributes))
    yvalues = np.zeros(observations)
    linenum = 0
    for line in data:
        temp = line.split(",")
        for i in range(0,numattributes):
                if i == yloc:
                    yvalues[linenum]=temp[i]
                else:
                    xvalues[linenum,i+1] = temp[i]
        linenum+=1
    return xvalues,yvalues;

def grabdataspace(filename,numattributes,observations,yloc):
    data=open(filename,'r')
    xvalues = np.ones((observations,numattributes))
    yvalues = np.zeros(observations)
    linenum = 0
    for line in data:
        temp = line.split(" ")
        for i in range(0,numattributes):
            if i == yloc:
                yvalues[linenum]=temp[i]
            else:
                xvalues[linenum,i+1] = temp[i]
        linenum+=1
    return xvalues,yvalues;

def pltmyreg(reg2score,reg4param,reg4score,yvalues):
    sortscores2=np.array(sorted(-reg2score))
    plt.figure(1,figsize=(20, 10))
    plt.subplot(212)
    plt.plot(-reg2score,yvalues,'bo',sortscores2,1/(1+np.exp(-sortscores2)),'k')
    #plt.ylim([min(yvalues)-min(.01,np.std(yvalues)),max(yvalues)+min(.01,np.std(yvalues))])
    #plt.xlim([min(-reg2score)-min(1,np.std(reg2score)),max(-reg2score)+min(1,np.std(reg2score))])
    plt.xlabel('Data with 2 Parameter Regression')
    plt.show()
    sortscores4=np.array(sorted(-reg4score))
    plt.figure(1,figsize=(20, 10))
    plt.subplot(212)
    plt.plot(-reg4score,yvalues,'bo',sortscores4,reg4param[0]+(reg4param[1]-reg4param[0])/(1+np.exp(-sortscores4)),'k')
    #plt.ylim([min(yvalues)-min(.01,np.std(yvalues)),max(yvalues)+min(.01,np.std(yvalues))])
    #plt.xlim([min(-reg4score)-min(1,np.std(reg4score)),max(-reg4score)+min(1,np.std(reg4score))])
    plt.xlabel('Data with 4 Parameter Regression')
    plt.show()
    return;