# Generated with SMOP  0.41
from libsmop import *
# gen_data.m

    
@function
def gen_data(M=None,N=None,*args,**kwargs):
    varargin = gen_data.varargin
    nargin = gen_data.nargin

    num_pop=5
# gen_data.m:3
    
    std=0.05
# gen_data.m:4
    R_idx=[]
# gen_data.m:5
    for u in arange(1,N).reshape(-1):
        pop=randi(concat([2,num_pop + 1]))
# gen_data.m:7
        pop_bias=dot(ceil(M / (num_pop + 2)),pop)
# gen_data.m:8
        pop_scale=dot(ceil(M / (num_pop + 2)),std)
# gen_data.m:9
        num_signals=randi(concat([10,20]))
# gen_data.m:10
        signals=round(dot(pop_scale,randn(num_signals,1)) + pop_bias)
# gen_data.m:12
        user_col=repmat(u,concat([length(signals),1]))
# gen_data.m:13
        R_idx=concat([[R_idx],[horzcat(user_col,signals)]])
# gen_data.m:14
    
    return R_idx
    
if __name__ == '__main__':
    pass
    