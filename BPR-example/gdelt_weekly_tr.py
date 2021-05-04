# Generated with SMOP  0.41
from libsmop import *
# gdelt_weekly_tr.m

    
@function
def gdelt_weekly_tr(base_path=None,week=None,reload=None,subset_size=None,*args,**kwargs):
    varargin = gdelt_weekly_tr.varargin
    nargin = gdelt_weekly_tr.nargin

    if reload == 1:
        # Get data
        train_path=strcat(base_path,'_',week,'_train.csv')
# gdelt_weekly_tr.m:5
        train_data=csvread(train_path,1,0)
# gdelt_weekly_tr.m:6
        # Test
        fd=fopen(strcat('data/source_map_',week,'.csv'))
# gdelt_weekly_tr.m:10
        line=fgets(fd)
# gdelt_weekly_tr.m:11
        names_train=strsplit(line,',')
# gdelt_weekly_tr.m:13
        fd=fopen(strcat('data/event_map_',week,'.csv'))
# gdelt_weekly_tr.m:15
        line=fgets(fd)
# gdelt_weekly_tr.m:16
        line=strsplit(line,',')
# gdelt_weekly_tr.m:17
        ids_train=str2double(line)
# gdelt_weekly_tr.m:18
    
    # Extract indices
    R_idx_data=concat([train_data(arange(),2),train_data(arange(),1)])
# gdelt_weekly_tr.m:22
    
    R_idx_tr=unique(R_idx_data,'rows')
# gdelt_weekly_tr.m:24
    R_idx_tr=datasample(R_idx_tr,subset_size)
# gdelt_weekly_tr.m:25
    
    extrema=max(R_idx_tr)
# gdelt_weekly_tr.m:27
    M_tr=extrema(2)
# gdelt_weekly_tr.m:28
    
    N_tr=extrema(1)
# gdelt_weekly_tr.m:29
    
    return R_idx_tr,M_tr,N_tr,ids_train,names_train
    
if __name__ == '__main__':
    pass
    