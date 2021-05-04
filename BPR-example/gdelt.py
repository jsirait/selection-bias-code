# Generated with SMOP  0.41
from libsmop import *
# gdelt.m

    
@function
def gdelt(path=None,subset_size=None,reload=None,*args,**kwargs):
    varargin = gdelt.varargin
    nargin = gdelt.nargin

    if reload == 1:
        # Get data
        data=csvread(path,1,0)
# gdelt.m:4
        fd=fopen('data/source_map.csv')
# gdelt.m:7
        line=fgets(fd)
# gdelt.m:8
        names=strsplit(line,',')
# gdelt.m:9
        fd=fopen('data/event_map.csv')
# gdelt.m:10
        line=fgets(fd)
# gdelt.m:11
        line=strsplit(line,',')
# gdelt.m:12
        ids=str2double(line)
# gdelt.m:13
    
    # Extract indices
    R_idx_data=concat([data(arange(),2),data(arange(),1)])
# gdelt.m:17
    
    R_idx_u=R_idx_data(arange(1,subset_size),arange())
# gdelt.m:19
    
    R_idx=unique(R_idx_u,'rows')
# gdelt.m:21
    
    extrema=max(R_idx)
# gdelt.m:23
    M=extrema(2)
# gdelt.m:24
    
    N=extrema(1)
# gdelt.m:25
    
    return R_idx,M,N,names,ids
    
if __name__ == '__main__':
    pass
    