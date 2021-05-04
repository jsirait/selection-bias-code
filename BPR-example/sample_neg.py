# Generated with SMOP  0.41
from libsmop import *
# sample_neg.m

    
@function
def sample_neg(R=None,u=None,*args,**kwargs):
    varargin = sample_neg.varargin
    nargin = sample_neg.nargin

    #SAMPLE_NEG sample an item that had no interaction with the given user
    while true:

        item=randi(concat([1,size(R,2)]))
# sample_neg.m:4
        if R(u,item) == 0:
            neg_i=copy(item)
# sample_neg.m:5
            break

    
    return neg_i
    
if __name__ == '__main__':
    pass
    