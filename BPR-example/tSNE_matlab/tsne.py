# Generated with SMOP  0.41
from libsmop import *
# tsne.m

    
@function
def tsne(X=None,labels=None,no_dims=None,initial_dims=None,perplexity=None,*args,**kwargs):
    varargin = tsne.varargin
    nargin = tsne.nargin

    #TSNE Performs symmetric t-SNE on dataset X
    
    #   mappedX = tsne(X, labels, no_dims, initial_dims, perplexity)
#   mappedX = tsne(X, labels, initial_solution, perplexity)
    
    # The function performs symmetric t-SNE on the NxD dataset X to reduce its 
# dimensionality to no_dims dimensions (default = 2). The data is 
# preprocessed using PCA, reducing the dimensionality to initial_dims 
# dimensions (default = 30). Alternatively, an initial solution obtained 
# from an other dimensionality reduction technique may be specified in 
# initial_solution. The perplexity of the Gaussian kernel that is employed 
# can be specified through perplexity (default = 30). The labels of the
# data are not used by t-SNE itself, however, they are used to color
# intermediate plots. Please provide an empty labels matrix [] if you
# don't want to plot results during the optimization.
# The low-dimensional data representation is returned in mappedX.
    
    
    # (C) Laurens van der Maaten, 2010
# University of California, San Diego
    
    if logical_not(exist('labels','var')):
        labels=[]
# tsne.m:25
    
    if logical_not(exist('no_dims','var')) or isempty(no_dims):
        no_dims=2
# tsne.m:28
    
    if logical_not(exist('initial_dims','var')) or isempty(initial_dims):
        initial_dims=min(50,size(X,2))
# tsne.m:31
    
    if logical_not(exist('perplexity','var')) or isempty(perplexity):
        perplexity=40
# tsne.m:34
    
    
    # First check whether we already have an initial solution
    if numel(no_dims) > 1:
        initial_solution=copy(true)
# tsne.m:39
        ydata=copy(no_dims)
# tsne.m:40
        no_dims=size(ydata,2)
# tsne.m:41
        perplexity=copy(initial_dims)
# tsne.m:42
    else:
        initial_solution=copy(false)
# tsne.m:44
    
    
    # Normalize input data
    X=X - min(ravel(X))
# tsne.m:48
    X=X / max(ravel(X))
# tsne.m:49
    X=bsxfun(minus,X,mean(X,1))
# tsne.m:50
    
    if logical_not(initial_solution):
        disp('Preprocessing data using PCA...')
        if size(X,2) < size(X,1):
            C=dot(X.T,X)
# tsne.m:56
        else:
            C=dot((1 / size(X,1)),(dot(X,X.T)))
# tsne.m:58
        M,lambda_=eig(C,nargout=2)
# tsne.m:60
        lambda_,ind=sort(diag(lambda_),'descend',nargout=2)
# tsne.m:61
        M=M(arange(),ind(arange(1,initial_dims)))
# tsne.m:62
        lambda_=lambda_(arange(1,initial_dims))
# tsne.m:63
        if logical_not((size(X,2) < size(X,1))):
            M=bsxfun(times,dot(X.T,M),(1 / sqrt(multiply(size(X,1),lambda_))).T)
# tsne.m:65
        X=dot(bsxfun(minus,X,mean(X,1)),M)
# tsne.m:67
        clear('M','lambda','ind')
    
    
    # Compute pairwise distance matrix
    sum_X=sum(X ** 2,2)
# tsne.m:72
    D=bsxfun(plus,sum_X,bsxfun(plus,sum_X.T,dot(- 2,(dot(X,X.T)))))
# tsne.m:73
    
    P=d2p(D,perplexity,1e-05)
# tsne.m:76
    
    clear('D')
    
    # Run t-SNE
    if initial_solution:
        ydata=tsne_p(P,labels,ydata)
# tsne.m:81
    else:
        ydata=tsne_p(P,labels,no_dims)
# tsne.m:83
    
    