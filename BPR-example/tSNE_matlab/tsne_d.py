# Generated with SMOP  0.41
from libsmop import *
# tsne_d.m

    
@function
def tsne_d(D=None,labels=None,no_dims=None,perplexity=None,*args,**kwargs):
    varargin = tsne_d.varargin
    nargin = tsne_d.nargin

    #TSNE_D Performs symmetric t-SNE on the pairwise Euclidean distance matrix D
    
    #   mappedX = tsne_d(D, labels, no_dims, perplexity)
#   mappedX = tsne_d(D, labels, initial_solution, perplexity)
    
    # The function performs symmetric t-SNE on the NxN pairwise Euclidean 
# distance matrix D to construct an embedding with no_dims dimensions 
# (default = 2). An initial solution obtained from an other dimensionality 
# reduction technique may be specified in initial_solution. 
# The perplexity of the Gaussian kernel that is employed can be specified 
# through perplexity (default = 30). The labels of the data are not used 
# by t-SNE itself, however, they are used to color intermediate plots. 
# Please provide an empty labels matrix [] if you don't want to plot 
# results during the optimization.
# The data embedding is returned in mappedX.
    
    
    # (C) Laurens van der Maaten, 2010
# University of California, San Diego
    
    if logical_not(exist('labels','var')):
        labels=[]
# tsne_d.m:24
    
    if logical_not(exist('no_dims','var')) or isempty(no_dims):
        no_dims=2
# tsne_d.m:27
    
    if logical_not(exist('perplexity','var')) or isempty(perplexity):
        perplexity=30
# tsne_d.m:30
    
    
    # First check whether we already have an initial solution
    if numel(no_dims) > 1:
        initial_solution=copy(true)
# tsne_d.m:35
        ydata=copy(no_dims)
# tsne_d.m:36
        no_dims=size(ydata,2)
# tsne_d.m:37
    else:
        initial_solution=copy(false)
# tsne_d.m:39
    
    
    # Compute joint probabilities
    D=D / max(ravel(D))
# tsne_d.m:43
    
    P=d2p(D ** 2,perplexity,1e-05)
# tsne_d.m:44
    
    
    # Run t-SNE
    if initial_solution:
        ydata=tsne_p(P,labels,ydata)
# tsne_d.m:48
    else:
        ydata=tsne_p(P,labels,no_dims)
# tsne_d.m:50
    
    