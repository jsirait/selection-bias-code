# Generated with SMOP  0.41
from libsmop import *
# gdelt_weekly_te.m

    
@function
def gdelt_weekly_te(base_path=None,week=None,reload=None,subset_size=None,*args,**kwargs):
    varargin = gdelt_weekly_te.varargin
    nargin = gdelt_weekly_te.nargin

    if reload == 1:
        # Get data
        test_path=strcat(base_path,'_',week,'_test.csv')
# gdelt_weekly_te.m:5
        test_data=csvread(test_path,1,0)
# gdelt_weekly_te.m:6
        # Test
        fd=fopen(strcat('data/source_map_',week,'.csv'))
# gdelt_weekly_te.m:10
        line=fgets(fd)
# gdelt_weekly_te.m:11
        names_test=strsplit(line,',')
# gdelt_weekly_te.m:13
        fd=fopen(strcat('data/event_map_',week,'.csv'))
# gdelt_weekly_te.m:15
        line=fgets(fd)
# gdelt_weekly_te.m:16
        line=strsplit(line,',')
# gdelt_weekly_te.m:17
        ids_test=str2double(line)
# gdelt_weekly_te.m:18
    
    # Extract indices
    R_idx_data=concat([test_data(arange(),2),test_data(arange(),1)])
# gdelt_weekly_te.m:22
    
    R_idx_te=unique(R_idx_data,'rows')
# gdelt_weekly_te.m:24
    R_idx_te=datasample(R_idx_te,subset_size)
# gdelt_weekly_te.m:25
    
    extrema=max(R_idx_te)
# gdelt_weekly_te.m:27
    M_te=extrema(2)
# gdelt_weekly_te.m:28
    
    N_te=extrema(1)
# gdelt_weekly_te.m:29
    
    return R_idx_te,M_te,N_te,ids_test,names_test
    
if __name__ == '__main__':
    pass
    