# Generated with SMOP  0.41
from libsmop import *
# x2p.m

    
@function
def x2p(X=None,u=None,tol=None,*args,**kwargs):
    varargin = x2p.varargin
    nargin = x2p.nargin

    #X2P Identifies appropriate sigma's to get kk NNs up to some tolerance
    
    #   [P, beta] = x2p(xx, kk, tol)
# 
# Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
# kernel with a certain uncertainty for every datapoint. The desired
# uncertainty can be specified through the perplexity u (default = 15). The
# desired perplexity is obtained up to some tolerance that can be specified
# by tol (default = 1e-4).
# The function returns the final Gaussian kernel in P, as well as the 
# employed precisions per instance in beta.
    
    
    # (C) Laurens van der Maaten, 2008
# Maastricht University
    
    
    if logical_not(exist('u','var')) or isempty(u):
        u=15
# x2p.m:20
    
    if logical_not(exist('tol','var')) or isempty(tol):
        tol=0.0001
# x2p.m:23
    
    
    # Initialize some variables
    n=size(X,1)
# x2p.m:27
    
    P=zeros(n,n)
# x2p.m:28
    
    beta=ones(n,1)
# x2p.m:29
    
    logU=log(u)
# x2p.m:30
    
    
    # Compute pairwise distances
    disp('Computing pairwise distances...')
    sum_X=sum(X ** 2,2)
# x2p.m:34
    D=bsxfun(plus,sum_X,bsxfun(plus,sum_X.T,dot(dot(- 2,X),X.T)))
# x2p.m:35
    
    disp('Computing P-values...')
    for i in arange(1,n).reshape(-1):
        if logical_not(rem(i,500)):
            disp(concat(['Computed P-values ',num2str(i),' of ',num2str(n),' datapoints...']))
        # Set minimum and maximum values for precision
        betamin=- Inf
# x2p.m:46
        betamax=copy(Inf)
# x2p.m:47
        Di=D(i,concat([arange(1,i - 1),arange(i + 1,end())]))
# x2p.m:50
        H,thisP=Hbeta(Di,beta(i),nargout=2)
# x2p.m:51
        Hdiff=H - logU
# x2p.m:54
        tries=0
# x2p.m:55
        while abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin=beta(i)
# x2p.m:60
                if isinf(betamax):
                    beta[i]=dot(beta(i),2)
# x2p.m:62
                else:
                    beta[i]=(beta(i) + betamax) / 2
# x2p.m:64
            else:
                betamax=beta(i)
# x2p.m:67
                if isinf(betamin):
                    beta[i]=beta(i) / 2
# x2p.m:69
                else:
                    beta[i]=(beta(i) + betamin) / 2
# x2p.m:71
            # Recompute the values
            H,thisP=Hbeta(Di,beta(i),nargout=2)
# x2p.m:76
            Hdiff=H - logU
# x2p.m:77
            tries=tries + 1
# x2p.m:78

        # Set the final row of P
        P[i,concat([arange(1,i - 1),arange(i + 1,end())])]=thisP
# x2p.m:82
    
    disp(concat(['Mean value of sigma: ',num2str(mean(sqrt(1 / beta)))]))
    disp(concat(['Minimum value of sigma: ',num2str(min(sqrt(1 / beta)))]))
    disp(concat(['Maximum value of sigma: ',num2str(max(sqrt(1 / beta)))]))
    return P,beta
    
if __name__ == '__main__':
    pass
    
    
    # Function that computes the Gaussian kernel values given a vector of
# squared Euclidean distances, and the precision of the Gaussian kernel.
# The function also computes the perplexity of the distribution.
    
@function
def Hbeta(D=None,beta=None,*args,**kwargs):
    varargin = Hbeta.varargin
    nargin = Hbeta.nargin

    P=exp(dot(- D,beta))
# x2p.m:95
    sumP=sum(P)
# x2p.m:96
    H=log(sumP) + dot(beta,sum(multiply(D,P))) / sumP
# x2p.m:97
    P=P / sumP
# x2p.m:98
    return H,P
    
if __name__ == '__main__':
    pass
    