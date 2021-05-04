# Generated with SMOP  0.41
from libsmop import *
# d2p.m

    
@function
def d2p(D=None,u=None,tol=None,*args,**kwargs):
    varargin = d2p.varargin
    nargin = d2p.nargin

    #D2P Identifies appropriate sigma's to get kk NNs up to some tolerance
    
    #   [P, beta] = d2p(D, kk, tol)
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
# d2p.m:20
    
    if logical_not(exist('tol','var')) or isempty(tol):
        tol=0.0001
# d2p.m:23
    
    
    # Initialize some variables
    n=size(D,1)
# d2p.m:27
    
    P=zeros(n,n)
# d2p.m:28
    
    beta=ones(n,1)
# d2p.m:29
    
    logU=log(u)
# d2p.m:30
    
    # Run over all datapoints
    for i in arange(1,n).reshape(-1):
        if logical_not(rem(i,500)):
            disp(concat(['Computed P-values ',num2str(i),' of ',num2str(n),' datapoints...']))
        # Set minimum and maximum values for precision
        betamin=- Inf
# d2p.m:40
        betamax=copy(Inf)
# d2p.m:41
        H,thisP=Hbeta(D(i,concat([arange(1,i - 1),arange(i + 1,end())])),beta(i),nargout=2)
# d2p.m:44
        Hdiff=H - logU
# d2p.m:47
        tries=0
# d2p.m:48
        while abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin=beta(i)
# d2p.m:53
                if isinf(betamax):
                    beta[i]=dot(beta(i),2)
# d2p.m:55
                else:
                    beta[i]=(beta(i) + betamax) / 2
# d2p.m:57
            else:
                betamax=beta(i)
# d2p.m:60
                if isinf(betamin):
                    beta[i]=beta(i) / 2
# d2p.m:62
                else:
                    beta[i]=(beta(i) + betamin) / 2
# d2p.m:64
            # Recompute the values
            H,thisP=Hbeta(D(i,concat([arange(1,i - 1),arange(i + 1,end())])),beta(i),nargout=2)
# d2p.m:69
            Hdiff=H - logU
# d2p.m:70
            tries=tries + 1
# d2p.m:71

        # Set the final row of P
        P[i,concat([arange(1,i - 1),arange(i + 1,end())])]=thisP
# d2p.m:75
    
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
# d2p.m:88
    sumP=sum(P)
# d2p.m:89
    H=log(sumP) + dot(beta,sum(multiply(D,P))) / sumP
# d2p.m:90
    
    P=P / sumP
# d2p.m:92
    return H,P
    
if __name__ == '__main__':
    pass
    