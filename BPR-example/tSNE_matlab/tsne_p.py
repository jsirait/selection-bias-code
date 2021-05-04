# Generated with SMOP  0.41
from libsmop import *
# tsne_p.m

    
@function
def tsne_p(P=None,labels=None,no_dims=None,*args,**kwargs):
    varargin = tsne_p.varargin
    nargin = tsne_p.nargin

    #TSNE_P Performs symmetric t-SNE on affinity matrix P
    
    #   mappedX = tsne_p(P, labels, no_dims)
    
    # The function performs symmetric t-SNE on pairwise similarity matrix P 
# to create a low-dimensional map of no_dims dimensions (default = 2).
# The matrix P is assumed to be symmetric, sum up to 1, and have zeros
# on the diagonal.
# The labels of the data are not used by t-SNE itself, however, they 
# are used to color intermediate plots. Please provide an empty labels
# matrix [] if you don't want to plot results during the optimization.
# The low-dimensional data representation is returned in mappedX.
    
    
    # (C) Laurens van der Maaten, 2010
# University of California, San Diego
    
    if logical_not(exist('labels','var')):
        labels=[]
# tsne_p.m:21
    
    if logical_not(exist('no_dims','var')) or isempty(no_dims):
        no_dims=2
# tsne_p.m:24
    
    
    # First check whether we already have an initial solution
    if numel(no_dims) > 1:
        initial_solution=copy(true)
# tsne_p.m:29
        ydata=copy(no_dims)
# tsne_p.m:30
        no_dims=size(ydata,2)
# tsne_p.m:31
    else:
        initial_solution=copy(false)
# tsne_p.m:33
    
    
    # Initialize some variables
    n=size(P,1)
# tsne_p.m:37
    
    momentum=0.5
# tsne_p.m:38
    
    final_momentum=0.8
# tsne_p.m:39
    
    mom_switch_iter=250
# tsne_p.m:40
    
    stop_lying_iter=100
# tsne_p.m:41
    
    max_iter=1000
# tsne_p.m:42
    
    epsilon=500
# tsne_p.m:43
    
    min_gain=0.01
# tsne_p.m:44
    
    
    # Make sure P-vals are set properly
    P[arange(1,end(),n + 1)]=0
# tsne_p.m:47
    
    P=dot(0.5,(P + P.T))
# tsne_p.m:48
    
    P=max(P / sum(ravel(P)),realmin)
# tsne_p.m:49
    
    const=sum(multiply(ravel(P),log(ravel(P))))
# tsne_p.m:50
    
    if logical_not(initial_solution):
        P=dot(P,4)
# tsne_p.m:52
    
    
    # Initialize the solution
    if logical_not(initial_solution):
        ydata=dot(0.0001,randn(n,no_dims))
# tsne_p.m:57
    
    y_incs=zeros(size(ydata))
# tsne_p.m:59
    gains=ones(size(ydata))
# tsne_p.m:60
    
    for iter in arange(1,max_iter).reshape(-1):
        # Compute joint probability that point i and j are neighbors
        sum_ydata=sum(ydata ** 2,2)
# tsne_p.m:66
        num=1 / (1 + bsxfun(plus,sum_ydata,bsxfun(plus,sum_ydata.T,dot(- 2,(dot(ydata,ydata.T))))))
# tsne_p.m:67
        num[arange(1,end(),n + 1)]=0
# tsne_p.m:68
        Q=max(num / sum(ravel(num)),realmin)
# tsne_p.m:69
        # Compute the gradients (faster implementation)
        L=multiply((P - Q),num)
# tsne_p.m:72
        y_grads=dot(dot(4,(diag(sum(L,1)) - L)),ydata)
# tsne_p.m:73
        gains=multiply((gains + 0.2),(sign(y_grads) != sign(y_incs))) + multiply((dot(gains,0.8)),(sign(y_grads) == sign(y_incs)))
# tsne_p.m:76
        gains[gains < min_gain]=min_gain
# tsne_p.m:78
        y_incs=dot(momentum,y_incs) - dot(epsilon,(multiply(gains,y_grads)))
# tsne_p.m:79
        ydata=ydata + y_incs
# tsne_p.m:80
        ydata=bsxfun(minus,ydata,mean(ydata,1))
# tsne_p.m:81
        if iter == mom_switch_iter:
            momentum=copy(final_momentum)
# tsne_p.m:85
        if iter == stop_lying_iter and logical_not(initial_solution):
            P=P / 4
# tsne_p.m:88
        # Print out progress
        if logical_not(rem(iter,10)):
            cost=const - sum(multiply(ravel(P),log(ravel(Q))))
# tsne_p.m:93
            disp(concat(['Iteration ',num2str(iter),': error is ',num2str(cost)]))
        # Display scatter plot (maximally first three dimensions)
        if logical_not(rem(iter,10)) and logical_not(isempty(labels)):
            if no_dims == 1:
                scatter(ydata,ydata,9,labels,'filled')
            else:
                if no_dims == 2:
                    scatter(ydata(arange(),1),ydata(arange(),2),9,labels,'filled')
                else:
                    scatter3(ydata(arange(),1),ydata(arange(),2),ydata(arange(),3),40,labels,'filled')
            axis('tight')
            axis('off')
            drawnow
    
    