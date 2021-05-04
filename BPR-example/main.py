# Generated with SMOP  0.41
from libsmop import *
# main.m

    # This code is a basic example of one-class Matrix Factorization
# using AUC as a ranking metric and Bayesian Personalized Ranking
# as an optimization procedure (https://arxiv.org/abs/1205.2618).
#clear;
    
    # TODO 
# * Cross validation
    
    set(0,'DefaultLineLineWidth',3)
    set(groot,'DefaultAxesFontSize',30)
    ##
    iter=20000000.0
# main.m:13
    
    alpha=0.1
# main.m:14
    
    lambda_=0.01
# main.m:15
    
    sigma=0.1
# main.m:16
    
    mu=0.0
# main.m:17
    
    K=20
# main.m:18
    
    reload=1
# main.m:19
    
    subset=1000000.0
# main.m:20
    
    tetr_ratio=0.2
# main.m:21
    
    path='data/hashed.csv'
# main.m:22
    
    week='W4'
# main.m:23
    ##
# M events
# N sources
# R_idx is an nx2 matrix holding the indices of positive signals
# names holds the string representation of sources
#[R_idx, M, N, names, ids] = gdelt(path, subset, reload);
    
    ## Create testing and training sets
    
    tetr_split=3
# main.m:33
    if tetr_split == 1:
        Rall=sparse(R_idx(arange(),1),R_idx(arange(),2),1)
# main.m:37
        idx_te=zeros(N,1) #junita: I think sparse is a matlab function
# main.m:38
        # per source
        for i in arange(1,N).reshape(-1):
            idxs=find(R_idx(arange(),1) == i)
# main.m:42
            rand_idx=randi(length(idxs),1)
# main.m:43
            idx_te[i]=idxs(rand_idx)
# main.m:44
        # Create index mask
    # Test
        test_mask=zeros(length(R_idx),1)
# main.m:49
        test_mask[idx_te]=1
# main.m:50
        test_mask=logical(test_mask)
# main.m:51
        train_mask=logical_not(test_mask)
# main.m:53
        R_idx_tr=R_idx(train_mask,arange())
# main.m:55
        R_idx_te=R_idx(test_mask,arange())
# main.m:56
        Rtr=sparse(R_idx_tr(arange(),1),R_idx_tr(arange(),2),1,N,M)
# main.m:58
        Rte=sparse(R_idx_te(arange(),1),R_idx_te(arange(),2),1,N,M)
# main.m:59
    else:
        if tetr_split == 2:
            Rall=sparse(R_idx(arange(),1),R_idx(arange(),2),1)
# main.m:62
            datalen=length(R_idx)
# main.m:63
            rp=randperm(datalen)
# main.m:64
            pivot=ceil(datalen / 10)
# main.m:65
            R_idx_te=R_idx(rp(arange(1,pivot)),arange())
# main.m:66
            R_idx_tr=R_idx(rp(arange(pivot + 1,end())),arange())
# main.m:67
            Rtr=sparse(R_idx_tr(arange(),1),R_idx_tr(arange(),2),1)
# main.m:70
        else:
            if tetr_split == 3:
                R_idx_te,M_te,N_te,ids_test,names_test=gdelt_weekly_te('data/hashed',week,reload,dot(subset,tetr_ratio),nargout=5)
# main.m:73
                R_idx_tr,M_tr,N_tr,ids_train,names_train=gdelt_weekly_tr('data/hashed',week,reload,dot(subset,(1 - tetr_ratio)),nargout=5)
# main.m:74
                names=copy(names_train)
# main.m:76
                M=max(M_te,M_tr)
# main.m:78
                N=max(N_te,N_tr)
# main.m:79
                R_idx=union(R_idx_te,R_idx_tr,'rows')
# main.m:81
                Rall=sparse(R_idx(arange(),1),R_idx(arange(),2),1)
# main.m:82
                idx_te=[]
# main.m:83
                # Leave one out per source
                for i in arange(1,N_te).reshape(-1):
                    idxs=find(R_idx_te(arange(),1) == i)
# main.m:87
                    if logical_not(isempty(idxs)):
                        rand_idx=randi(length(idxs),1)
# main.m:89
                        idx_te=concat([[idx_te],[idxs(rand_idx)]])
# main.m:90
                # Keep only the heldout test samples
    # Create a mask
                not_idx_te=zeros(length(R_idx_te),1)
# main.m:96
                not_idx_te[idx_te]=true
# main.m:97
                not_idx_te=logical_not(not_idx_te)
# main.m:98
                R_idx_tr=concat([[R_idx_tr],[R_idx_te(not_idx_te,arange())]])
# main.m:100
                R_idx_te[not_idx_te,arange()]=[]
# main.m:101
                # heldout samples
                # Create the Source-Event interaction matrix
                Rtr=sparse(R_idx_tr(arange(),1),R_idx_tr(arange(),2),1,N,M)
# main.m:105
                Rte=sparse(R_idx_te(arange(),1),R_idx_te(arange(),2),1,N,M)
# main.m:106
    
    # Sanity checks (nnz elements of Rall should be equal to the number of
# indices provided
    if length(R_idx) != nnz(Rall) and tetr_split != 3:
        disp('Problem in Rall.')
    else:
        if length(union(R_idx_te,R_idx_tr,'rows')) != nnz(Rall) and tetr_split == 3:
            disp('Problen in Rall (tetr==3)')
            disp(length(R_idx_tr) + length(R_idx_te) - nnz(Rall))
    
    ## Run BPR
    
    # Record auc values
    auc_vals=zeros(iter / 100000,1)
# main.m:122
    # Initialize low-rank matrices with random values
    P=multiply(sigma,randn(N,K)) + mu
# main.m:125
    
    Q=multiply(sigma,randn(K,M)) + mu
# main.m:126
    
    for step in arange(1,iter).reshape(-1):
        # Select a random positive example
        i=randi(concat([1,length(R_idx_tr)]))
# main.m:131
        iu=R_idx_tr(i,1)
# main.m:132
        ii=R_idx_tr(i,2)
# main.m:133
        ji=sample_neg(Rtr,iu)
# main.m:136
        px=(dot(P(iu,arange()),(Q(arange(),ii) - Q(arange(),ji))))
# main.m:139
        z=1 / (1 + exp(px))
# main.m:140
        d=dot((Q(arange(),ii) - Q(arange(),ji)),z) - dot(lambda_,P(iu,arange()).T)
# main.m:143
        P[iu,arange()]=P(iu,arange()) + dot(alpha,d.T)
# main.m:144
        d=dot(P(iu,arange()),z) - dot(lambda_,Q(arange(),ii).T)
# main.m:147
        Q[arange(),ii]=Q(arange(),ii) + dot(alpha,d.T)
# main.m:148
        d=dot(- P(iu,arange()),z) - dot(lambda_,Q(arange(),ji).T)
# main.m:151
        Q[arange(),ji]=Q(arange(),ji) + dot(alpha,d.T)
# main.m:152
        if mod(step,100000) == 0:
            # Compute the Area Under the Curve (AUC)
            auc=0
# main.m:157
            for i in arange(1,length(R_idx_te)).reshape(-1):
                te_i=randi(concat([1,length(R_idx_te)]))
# main.m:159
                te_iu=R_idx_te(i,1)
# main.m:160
                te_ii=R_idx_te(i,2)
# main.m:161
                te_ji=sample_neg(Rall,te_iu)
# main.m:162
                sp=dot(P(te_iu,arange()),Q(arange(),te_ii))
# main.m:164
                sn=dot(P(te_iu,arange()),Q(arange(),te_ji))
# main.m:165
                if sp > sn:
                    auc=auc + 1
# main.m:167
                else:
                    if sp == sn:
                        auc=auc + 0.5
# main.m:167
            auc=auc / length(R_idx_te)
# main.m:169
            fprintf(concat(['AUC test: ',num2str(auc),'\n']))
            auc_vals[step / 100000]=auc
# main.m:171
    
    ## t-SNE for users' latent factors - Computation
    
    addpath('tSNE_matlab/')
    plot_top_20=1
# main.m:180
    
    plot_names=1
# main.m:181
    
    plot_subset=arange(1,1000)
# main.m:182
    
    # Get index of top 1K sources
    __,I=sort(sum(Rall,2),1,'descend',nargout=2)
# main.m:185
    subidx=I(plot_subset)
# main.m:186
    ##
# Run t-SNE on subset
    ydata=tsne(P(subidx,arange()))
# main.m:190
    ##
#t-SNE for users' latent factors - Plot
    
    # Get ids for known sources to show them in plot
    if plot_top_20 == 1:
        top_20_str=cellarray(['cnn.com','bbc.com','nytimes.com','foxnews.com','washingtonpost.com','usatoday.com','theguardian.com','dailymail.co.uk','chinadaily.com.cn','telegraph.co.uk','wsj.com','indiatimes.com','independent.co.uk','elpais.com','lemonde.fr','ft.com','bostonglobe.com','ap.org','afp.com','reuters.com','yahoo.com'])
# main.m:197
        top_right_str=cellarray(['cbn.com','breitbart.com','spectator.org','foxnews.com','nypost.com','nationalreview.com','newsmax.com'])
# main.m:204
        top_left_str=cellarray(['democracynow.org','huffingtonpost.com','motherjones.com','newrepublic.com','salon.com','time.com'])
# main.m:205
        top_str=cellarray(['cbn.com','breitbart.com','spectator.org','foxnews.com','nypost.com','nationalreview.com','newsmax.com','democracynow.org','huffingtonpost.com','motherjones.com','newrepublic.com','salon.com','time.com','cnn.com','bbc.com','nytimes.com','foxnews.com','washingtonpost.com','usatoday.com','theguardian.com','dailymail.co.uk','chinadaily.com.cn','telegraph.co.uk','wsj.com','indiatimes.com','independent.co.uk','elpais.com','lemonde.fr','ft.com','bostonglobe.com','ap.org','afp.com','reuters.com','yahoo.com'])
# main.m:206
        top_20_ids=zeros(length(top_20_str),1)
# main.m:213
        top_right_ids=zeros(length(top_right_str),1)
# main.m:215
        top_left_ids=zeros(length(top_left_str),1)
# main.m:216
        top_ids=zeros(length(top_str),1)
# main.m:217
        for ii in arange(1,length(top_20_str)).reshape(-1):
            id_find=find(strcmp(top_20_str[ii],names_train))
# main.m:220
            if length(id_find) > 0:
                top_20_ids[ii]=id_find
# main.m:222
        for ii in arange(1,length(top_right_str)).reshape(-1):
            id_find=find(strcmp(top_right_str[ii],names_train))
# main.m:227
            if length(id_find) > 0:
                top_right_ids[ii]=id_find
# main.m:229
        for ii in arange(1,length(top_left_str)).reshape(-1):
            id_find=find(strcmp(top_left_str[ii],names_train))
# main.m:234
            if length(id_find) > 0:
                top_left_ids[ii]=id_find
# main.m:236
        for ii in arange(1,length(top_str)).reshape(-1):
            id_find=find(strcmp(top_str[ii],names_train))
# main.m:241
            if length(id_find) > 0:
                top_ids[ii]=id_find
# main.m:243
        top_20_ids=top_20_ids(top_20_ids > 0)
# main.m:247
        top_left_ids=top_left_ids(top_left_ids > 0)
# main.m:248
        top_right_ids=top_right_ids(top_right_ids > 0)
# main.m:249
        top_ids=top_ids(top_ids > 0)
# main.m:250
        plot_idx=ismember(subidx,top_20_ids)
# main.m:252
        # plot_idx = ismember(subidx,top_right_ids);
    # plot_idx = ismember(subidx,top_left_ids);
    # plot_idx = ismember(subidx,top_ids);
    # ydata = tsne(P(plot_idx,:));
    else:
        plot_idx=copy(subidx)
# main.m:258
    
    # Scatter plot t-SNE results
# figure;
# scatter(ydata(~plot_idx,1),ydata(~plot_idx,2));
# hold on;
# scatter(ydata(plot_idx,1),ydata(plot_idx,2), 300, 'r', 'filled');
    
    figure
    set(gca,'FontSize',30)
    scatter(ydata(logical_not(plot_idx),1),ydata(logical_not(plot_idx),2),'MarkerEdgeColor',concat([0,0.5,0.5]),'MarkerFaceColor',concat([0,0.7,0.7]),'LineWidth',1.5)
    hold('on')
    scatter(ydata(plot_idx,1),ydata(plot_idx,2),300,'MarkerEdgeColor',concat([0.5,0,0]),'MarkerFaceColor',concat([0.9,0,0]),'LineWidth',1.5)
    plot_names=2
# main.m:279
    # Overlay names
    if plot_names == 1:
        dx=0.75
# main.m:283
        dy=0.1
# main.m:283
        t=text(ydata(plot_idx,1) + dx,ydata(plot_idx,2) + dy,names_train(subidx(plot_idx)))
# main.m:284
        set(t,'FontSize',30)
        set(t,'FontWeight','bold')
    else:
        dx=0.1
# main.m:288
        dy=0.1
# main.m:288
        t=text(ydata(arange(),1) + dx,ydata(arange(),2) + dy,names_train(subidx))
# main.m:289
        set(t,'FontSize',30)
    
    xlabel('PC1')
    ylabel('PC2')
    title('t-SNE projection sources latent space P')
    hold('off')
    ## Plot Distance to Reuters + AP
    
    # Reuters
    reuters_id=find(strcmp('reuters.com',names_train))
# main.m:303
    reuters=Rall(reuters_id,arange())
# main.m:304
    reuters_idx=find(subidx == reuters_id)
# main.m:305
    # Associated Press
    ap_id=find(strcmp('ap.org',names_train))
# main.m:308
    ap=Rall(ap_id,arange())
# main.m:309
    ap_idx=find(subidx == ap_id)
# main.m:310
    # Compute distance
    dist=lambda id=None,source=None: log(nnz(logical_and(source,Rall(id,arange()))) / sum(source))
# main.m:313
    recompute_dist=1
# main.m:315
    if recompute_dist == 1:
        dist_reuters=zeros(1,length(subidx))
# main.m:318
        dist_ap=zeros(1,length(subidx))
# main.m:319
        for i in arange(1,length(subidx)).reshape(-1):
            source=Rall(subidx(i),arange())
# main.m:322
            dist_reuters[i]=dist(reuters_id,source)
# main.m:323
            dist_ap[i]=dist(ap_id,source)
# main.m:324
    
    # Plot
    figure
    scatter(ydata(arange(),1),ydata(arange(),2),100,dist_ap,'filled')
    hold('on')
    # Scatter
    scatter(ydata(reuters_idx,1),ydata(reuters_idx,2),300,'r','filled')
    scatter(ydata(ap_idx,1),ydata(ap_idx,2),300,'r','filled')
    # Overlay names
    t1=text(ydata(reuters_idx,1) + dx,ydata(reuters_idx,2) + dy,'Reuters')
# main.m:338
    t2=text(ydata(ap_idx,1) + dx,ydata(ap_idx,2) + dy,'Associated Press')
# main.m:339
    set(t1,'FontSize',30)
    set(t2,'FontSize',30)
    set(t1,'FontWeight','bold')
    set(t2,'FontWeight','bold')
    colorbar
    xlabel('PC1')
    ylabel('PC2')
    title('Log-Distance of each source to Associated Press')
    figure
    scatter(ydata(arange(),1),ydata(arange(),2),100,dist_reuters,'filled')
    hold('on')
    # Scatter
    scatter(ydata(reuters_idx,1),ydata(reuters_idx,2),300,'r','filled')
    scatter(ydata(ap_idx,1),ydata(ap_idx,2),300,'r','filled')
    # Overlay names
    t1=text(ydata(reuters_idx,1) + dx,ydata(reuters_idx,2) + dy,'Reuters')
# main.m:357
    t2=text(ydata(ap_idx,1) + dx,ydata(ap_idx,2) + dy,'Associated Press')
# main.m:358
    set(t1,'FontSize',30)
    set(t2,'FontSize',30)
    set(t1,'FontWeight','bold')
    set(t2,'FontWeight','bold')
    colorbar
    xlabel('PC1')
    ylabel('PC2')
    title('Log-Distance of each source to Reuters')
    ## DBSCAN - Copyright (c) 2015, Yarpiz
    
    addpath('DBSCAN/')
    # Configure
    epsilon=2
# main.m:373
    MinPts=5
# main.m:374
    X=copy(ydata)
# main.m:375
    # Compute
    db=DBSCAN(X,epsilon,MinPts)
# main.m:377
    # Plot
    PlotClusterinResult(X,db)
    ## Find recommendation ranking for holdout test event
# Manually curated top_20
    
    for i in arange(1,length(top_20_ids)).reshape(-1):
        search=top_20_ids(i)
# main.m:386
        if search < N:
            names_test(search)
            # dot product : P(i) . Q
            C=sum(bsxfun(times,P(search,arange()),Q.T),2)
# main.m:390
            tr_idx=find(Rtr(search,arange()))
# main.m:392
            C[tr_idx]=- 1000
# main.m:393
            __,I_d=sort(C,1,'descend',nargout=2)
# main.m:395
            holdout_event=find(Rte(search,arange()))
# main.m:397
            holdout_event_id=holdout_event(1)
# main.m:398
            global_id=ids_test(holdout_event_id) + 1
# main.m:399
            # Find its ranking
            ranking=find(I_d == holdout_event_id)
# main.m:401
    
    ## Find recommendation ranking for holdout test event
# top_20 from the dataset
    
    auto_top_20_ids=subidx(arange(1,20))
# main.m:408
    for i in arange(1,length(auto_top_20_ids)).reshape(-1):
        search=auto_top_20_ids(i)
# main.m:411
        if search < N:
            names_test(search)
            # dot product : P(i) . Q
            C=sum(bsxfun(times,P(search,arange()),Q.T),2)
# main.m:415
            tr_idx=find(Rtr(search,arange()))
# main.m:417
            C[tr_idx]=- 1000
# main.m:418
            __,I_d=sort(C,1,'descend',nargout=2)
# main.m:420
            holdout_event=find(Rte(search,arange()))
# main.m:422
            if numel(holdout_event) > 0:
                holdout_event_id=holdout_event(1)
# main.m:424
                global_id=ids_test(holdout_event_id) + 1
# main.m:425
                # Find its ranking
                ranking=find(I_d == holdout_event_id)
# main.m:427
            else:
                'No holdout found'
    
    ## Ranking - Jay
# Alternative implementation of the recommendation ranking system
    
    # Choose a subset to rank
    auto_top_20_ids=subidx(arange(1,end()))
# main.m:439
    
    unique_te=unique(R_idx_te(arange(),2))
# main.m:440
    
    res=[]
# main.m:442
    for i in arange(1,length(R_idx_te)).reshape(-1):
        te_ev=R_idx_te(i,arange())
# main.m:444
        sp=dot(P(te_ev(1),arange()),Q(arange(),te_ev(2)))
# main.m:445
        if any(te_ev(1) == auto_top_20_ids):
            cnt=1
# main.m:448
            for j in arange(1,length(unique_te)).reshape(-1):
                if i == j:
                    continue
                sn=dot(P(te_ev(1),arange()),Q(arange(),R_idx_te(j,2)))
# main.m:451
                # all the events
                if sn > sp:
                    cnt=cnt + 1
# main.m:454
                # for another event than
                                                # the one we had selected
            res=concat([[res],[cnt]])
# main.m:458
    
    ## Ranking - Popularity
    
    __,I=sort(sum(Rall.T,2),1,'descend',nargout=2)
# main.m:464
    subidx_events=I(arange(1,1000))
# main.m:465
    ##
# Choose a subset to rank
    auto_top_20_ids=subidx(arange(1,end()))
# main.m:468
    
    unique_te=unique(R_idx_te(arange(),2))
# main.m:469
    
    res_pop=[]
# main.m:471
    for i in arange(1,length(R_idx_te)).reshape(-1):
        te_ev=R_idx_te(i,arange())
# main.m:473
        sp=find(I == te_ev(2))
# main.m:474
        if any(te_ev(1) == auto_top_20_ids):
            res_pop=concat([[res_pop],[sp]])
# main.m:476
    
    ## Popularity AUC
    
    # Record auc values
    auc_vals_pop=zeros(iter / 100000,1)
# main.m:483
    for step in arange(1,iter).reshape(-1):
        # Select a random positive example
        i=randi(concat([1,length(R_idx_tr)]))
# main.m:488
        iu=R_idx_tr(i,1)
# main.m:489
        ii=R_idx_tr(i,2)
# main.m:490
        ji=sample_neg(Rtr,iu)
# main.m:493
        if mod(step,100000) == 0:
            # Compute the Area Under the Curve (AUC)
            auc=0
# main.m:498
            for i in arange(1,length(R_idx_te)).reshape(-1):
                te_i=randi(concat([1,length(R_idx_te)]))
# main.m:500
                te_iu=R_idx_te(i,1)
# main.m:501
                te_ii=R_idx_te(i,2)
# main.m:502
                te_ji=sample_neg(Rall,te_iu)
# main.m:503
                sp=sum(Rtr(arange(),te_ii))
# main.m:505
                sn=sum(Rtr(arange(),te_ji))
# main.m:506
                if sp > sn:
                    auc=auc + 1
# main.m:508
                else:
                    if sp == sn:
                        auc=auc + 0.5
# main.m:508
            auc=auc / length(R_idx_te)
# main.m:510
            fprintf(concat(['AUC test: ',num2str(auc),'\n']))
            auc_vals_pop[step / 100000]=auc
# main.m:512
    
    ## Top 20 dustribution
    
    auto_top_20_ids=subidx(arange(1,50))
# main.m:519
    
    unique_te=unique(R_idx_te(arange(),2))
# main.m:520
    
    res_20=[]
# main.m:522
    for i in arange(1,length(R_idx_te)).reshape(-1):
        te_ev=R_idx_te(i,arange())
# main.m:524
        sp=dot(P(te_ev(1),arange()),Q(arange(),te_ev(2)))
# main.m:525
        if any(te_ev(1) == auto_top_20_ids):
            cnt=1
# main.m:528
            for j in arange(1,length(unique_te)).reshape(-1):
                if i == j:
                    continue
                sn=dot(P(te_ev(1),arange()),Q(arange(),R_idx_te(j,2)))
# main.m:531
                # all the events
                if sn > sp:
                    cnt=cnt + 1
# main.m:534
                # for another event than
                                                # the one we had selected
            res_20=concat([[res_20],[cnt]])
# main.m:538
    
    ## Ranking plot
    
    ranks=copy(res)
# main.m:544
    figure
    h=hist(res,500)
# main.m:546
    scatter(arange(1,length(h),1),h,100,'MarkerEdgeColor',concat([0,0.5,0.5]),'MarkerFaceColor',concat([0,0.7,0.7]),'LineWidth',1.5)
    hold('on')
    h1=hist(res_pop,500)
# main.m:552
    scatter(arange(1,length(h1),1),h1,100,'MarkerEdgeColor',concat([0.5,0,0.5]),'MarkerFaceColor',concat([0.5,0,0.7]),'LineWidth',1.5)
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    ylabel('Count')
    xlabel('Ranking')
    title('Event ranking distribution')
    grid('on')
    #set(gca, 'XTickLabel', num2str([1:1:50, 100:100:500, 1000:1000:2000]))
    
    ## Popularity plot
    
    # popularity = f(#event)
    
    a=sum(Rall.T,2)
# main.m:569
    figure
    h1=hist(a,unique(a))
# main.m:571
    scatter(arange(1,length(h1),1),h1,100,'MarkerEdgeColor',concat([0,0.5,0.5]),'MarkerFaceColor',concat([0,0.5,0.7]),'LineWidth',1.5)
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    ylabel('# Events')
    xlabel('Popularity')
    title('Event popularity distribution')
    grid('on')
    ##
    
    a=sum(Rall,2)
# main.m:585
    figure
    h1=hist(a,unique(a))
# main.m:587
    scatter(arange(1,length(h1),1),h1,100,'MarkerEdgeColor',concat([0.5,0,0.5]),'MarkerFaceColor',concat([0.5,0,0.7]),'LineWidth',1.5)
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    ylabel('# Sources')
    xlabel('Events covered')
    title('Source coverage distribution')
    grid('on')
    ## Sanity check - Jay
# Check AUC score consistency
    
    unique_te=unique(R_idx_te(arange(),2))
# main.m:602
    
    auc=0
# main.m:604
    for i in arange(1,length(R_idx_te)).reshape(-1):
        te_ev=R_idx_te(i,arange())
# main.m:606
        sp=dot(P(te_ev(1),arange()),Q(arange(),te_ev(2)))
# main.m:607
        te_i=te_ev(2)
# main.m:609
        while te_i == te_ev(2):

            rand_i=randi(concat([1,length(R_idx_te)]))
# main.m:612
            te_i=R_idx_te(rand_i,2)
# main.m:613

        sn=dot(P(te_ev(1),arange()),Q(arange(),te_i))
# main.m:616
        if sp > sn:
            auc=auc + 1
# main.m:617
        else:
            if sp == sn:
                auc=auc + 0.5
# main.m:617
    
    auc=auc / length(R_idx_te)
# main.m:621
    
    fprintf(concat(['AUC test: ',num2str(auc),'\n']))
    ##
    alphas=concat([0.001,0.01,0.05,0.1,0.5,1])
# main.m:625
    Ks=concat([2,5,10,20,30,50])
# main.m:626
    figure
    colormap('default')
    imagesc(heatmap)
    ylabel('Learning rate (\alpha)')
    xlabel('Latent factors (K)')
    set(gca,'XTickLabel',Ks)
    set(gca,'YTickLabel',alphas)
    title('AUC (2e7 iterations, 91421 observations, 5970 holdout)')
    colorbar
    ## CV
    
    R_idx_te,M_te,N_te,ids_test,names_test=gdelt_weekly_te('data/hashed',reload,dot(subset,tetr_ratio),nargout=5)
# main.m:641
    R_idx_tr,M_tr,N_tr,ids_train,names_train=gdelt_weekly_tr('data/hashed',reload,dot(subset,(1 - tetr_ratio)),nargout=5)
# main.m:642
    names=copy(names_train)
# main.m:644
    M=max(M_te,M_tr)
# main.m:646
    N=max(N_te,N_tr)
# main.m:647
    R_idx=union(R_idx_te,R_idx_tr,'rows')
# main.m:649
    Rall=sparse(R_idx(arange(),1),R_idx(arange(),2),1)
# main.m:650
    idx_te=[]
# main.m:651
    
    # Leave one out per source
    for i in arange(1,N_te).reshape(-1):
        idxs=find(R_idx_te(arange(),1) == i)
# main.m:655
        if logical_not(isempty(idxs)):
            rand_idx=randi(length(idxs),1)
# main.m:657
            idx_te=concat([[idx_te],[idxs(rand_idx)]])
# main.m:658
    
    # Keep only the heldout test samples
# Create a mask
    not_idx_te=zeros(length(R_idx_te),1)
# main.m:664
    not_idx_te[idx_te]=true
# main.m:665
    not_idx_te=logical_not(not_idx_te)
# main.m:666
    R_idx_tr=concat([[R_idx_tr],[R_idx_te(not_idx_te,arange())]])
# main.m:668
    
    R_idx_te[not_idx_te,arange()]=[]
# main.m:669
    
    # heldout samples
    
    # Create the Source-Event interaction matrix
    Rtr=sparse(R_idx_tr(arange(),1),R_idx_tr(arange(),2),1,N,M)
# main.m:673
    Rte=sparse(R_idx_te(arange(),1),R_idx_te(arange(),2),1,N,M)
# main.m:674
    ##
    iter=10000000.0
# main.m:676
    alpha=0.1
# main.m:677
    lambdas=concat([[0.0001],[0.001],[0.01],[0.1],[0.5],[1]])
# main.m:679
    Ks=concat([[2],[5],[10],[20],[30],[50]])
# main.m:680
    auc_cv=zeros(length(lambdas),length(Ks))
# main.m:682
    for cv_iter_lambdas in arange(1,length(lambdas)).reshape(-1):
        for cv_iter_ks in arange(1,length(Ks)).reshape(-1):
            # Record auc values
            auc_vals=zeros(iter / 100000,1)
# main.m:688
            P=multiply(sigma,randn(N,Ks(cv_iter_ks))) + mu
# main.m:691
            Q=multiply(sigma,randn(Ks(cv_iter_ks),M)) + mu
# main.m:692
            for step in arange(1,iter).reshape(-1):
                # Select a random positive example
                i=randi(concat([1,length(R_idx_tr)]))
# main.m:697
                iu=R_idx_tr(i,1)
# main.m:698
                ii=R_idx_tr(i,2)
# main.m:699
                ji=sample_neg(Rtr,iu)
# main.m:702
                px=(dot(P(iu,arange()),(Q(arange(),ii) - Q(arange(),ji))))
# main.m:705
                z=1 / (1 + exp(px))
# main.m:706
                d=dot((Q(arange(),ii) - Q(arange(),ji)),z) - dot(lambdas(cv_iter_lambdas),P(iu,arange()).T)
# main.m:709
                P[iu,arange()]=P(iu,arange()) + dot(alpha,d.T)
# main.m:710
                d=dot(P(iu,arange()),z) - dot(lambdas(cv_iter_lambdas),Q(arange(),ii).T)
# main.m:713
                Q[arange(),ii]=Q(arange(),ii) + dot(alpha,d.T)
# main.m:714
                d=dot(- P(iu,arange()),z) - dot(lambdas(cv_iter_lambdas),Q(arange(),ji).T)
# main.m:717
                Q[arange(),ji]=Q(arange(),ji) + dot(alpha,d.T)
# main.m:718
                if mod(step,100000) == 0:
                    # Compute the Area Under the Curve (AUC)
                    auc=0
# main.m:723
                    for i in arange(1,length(R_idx_te)).reshape(-1):
                        te_i=randi(concat([1,length(R_idx_te)]))
# main.m:725
                        te_iu=R_idx_te(i,1)
# main.m:726
                        te_ii=R_idx_te(i,2)
# main.m:727
                        te_ji=sample_neg(Rall,te_iu)
# main.m:728
                        sp=dot(P(te_iu,arange()),Q(arange(),te_ii))
# main.m:730
                        sn=dot(P(te_iu,arange()),Q(arange(),te_ji))
# main.m:731
                        if sp > sn:
                            auc=auc + 1
# main.m:733
                        else:
                            if sp == sn:
                                auc=auc + 0.5
# main.m:733
                    auc=auc / length(R_idx_te)
# main.m:735
                    fprintf(concat(['AUC test: ',num2str(auc),'\n']))
                    auc_vals[step / 100000]=auc
# main.m:737
            auc_cv[cv_iter_lambdas,cv_iter_ks]=max(auc_vals)
# main.m:742
    
    ## CV heatmap
    
    figure
    colormap('default')
    imagesc(auc_cv)
    ylabel('Regularization (\lambda)')
    xlabel('Latent factors (K)')
    set(gca,'XTickLabel',Ks)
    set(gca,'YTickLabel',lambdas)
    title('AUC (2e7 iterations, 91421 observations, 5968 holdout -- \alpha = 0.1)')
    colorbar
    ## AUC plot
    
    figure
    xs=concat([arange(1,20000000.0,100000.0)])
# main.m:761
    ys=copy(auc_vals)
# main.m:762
    plot(xs,ys,'LineWidth',2.5)
    hold('on')
    plot(xs,multiply(ones(1,length(xs)),max(auc_vals)),'--','LineWidth',2.5)
    plot(xs,auc_vals_pop)
    grid('on')
    ylabel('AUC')
    xlabel('Iteration')
    legend('AUC','max(AUC)')
    title('AUC (2e7 iterations, \alpha=0.1, \lambda=0.01, K=20)')
    ## KNN popularity
    
    # Record auc values
    auc_vals_knn=zeros(iter / 100000,1)
# main.m:776
    # Compute the Area Under the Curve (AUC)
    auc=0
# main.m:779
    r=randi(concat([1,length(R_idx_te)]),1,1000)
# main.m:780
    for i in arange(1,length(r)).reshape(-1):
        i
        te_i=randi(concat([1,length(R_idx_te)]))
# main.m:784
        te_iu=R_idx_te(r(i),1)
# main.m:785
        te_ii=R_idx_te(r(i),2)
# main.m:786
        te_ji=sample_neg(Rall,te_iu)
# main.m:787
        knn_k=10
# main.m:789
        n,d=knnsearch(Rtr(concat([arange(1,te_iu - 1),arange(te_iu + 1,end())]),arange()),Rtr(te_iu,arange()),'k',knn_k,'distance','jaccard',nargout=2)
# main.m:791
        sp=sum(Rtr(n,te_ii))
# main.m:793
        sn=sum(Rtr(n,te_ji))
# main.m:794
        if sp > sn:
            auc=auc + 1
# main.m:797
        else:
            if sp == sn:
                auc=auc + 0.5
# main.m:797
    
    auc=auc / length(r)
# main.m:799
    fprintf(concat(['AUC test: ',num2str(auc),'\n']))
    ## Baseline run
    
    # Get index of top 1K sources
    __,I=sort(sum(Rall,2),1,'descend',nargout=2)
# main.m:805
    subidx=I(plot_subset)
# main.m:806
    # Record auc values
    auc_vals_pop=zeros(iter / 100000,1)
# main.m:809
    for step in arange(1,5).reshape(-1):
        # Select a random positive example
        i=randi(concat([1,length(R_idx_tr)]))
# main.m:814
        iu=R_idx_tr(i,1)
# main.m:815
        ii=R_idx_tr(i,2)
# main.m:816
        ji=sample_neg(Rtr,iu)
# main.m:819
        auc=0
# main.m:822
        for i in arange(1,length(R_idx_te)).reshape(-1):
            te_i=randi(concat([1,length(R_idx_te)]))
# main.m:824
            te_iu=R_idx_te(i,1)
# main.m:825
            te_ii=R_idx_te(i,2)
# main.m:826
            te_ji=sample_neg(Rall,te_iu)
# main.m:827
            sp=sum(Rtr(arange(),te_ii))
# main.m:829
            sn=sum(Rtr(arange(),te_ji))
# main.m:830
            if sp > sn:
                auc=auc + 1
# main.m:832
            else:
                if sp == sn:
                    auc=auc + 0.5
# main.m:832
        auc=auc / length(R_idx_te)
# main.m:834
        fprintf(concat(['AUC test: ',num2str(auc),'\n']))
        auc_vals_pop[step]=auc
# main.m:836
    
    # Record auc values
    auc_vals_knn=zeros(iter / 100000,1)
# main.m:842
    # Compute the Area Under the Curve (AUC)
    auc=0
# main.m:845
    r=randi(concat([1,length(R_idx_te)]),1,50)
# main.m:846
    for i in arange(1,length(r)).reshape(-1):
        i
        te_i=randi(concat([1,length(R_idx_te)]))
# main.m:850
        te_iu=R_idx_te(r(i),1)
# main.m:851
        te_ii=R_idx_te(r(i),2)
# main.m:852
        te_ji=sample_neg(Rall,te_iu)
# main.m:853
        knn_k=10
# main.m:855
        n,d=knnsearch(Rtr(concat([arange(1,te_iu - 1),arange(te_iu + 1,end())]),arange()),Rtr(te_iu,arange()),'k',knn_k,'distance','jaccard',nargout=2)
# main.m:857
        sp=sum(Rtr(n,te_ii))
# main.m:859
        sn=sum(Rtr(n,te_ji))
# main.m:860
        if sp > sn:
            auc=auc + 1
# main.m:863
        else:
            if sp == sn:
                auc=auc + 0.5
# main.m:863
    
    auc=auc / length(r)
# main.m:865
    fprintf(concat(['AUC test: ',num2str(auc),'\n']))
    auc_vals_knn=copy(auc)
# main.m:867
    ##
    
    load('../../clustering/W1/sources.csv')
    ydata=concat([sources(arange(),3),sources(arange(),4)])
# main.m:872
    subidx=sources(arange(),1)
# main.m:874
    # Scatter plot t-SNE results
# figure;
# scatter(ydata(~plot_idx,1),ydata(~plot_idx,2));
# hold on;
# scatter(ydata(plot_idx,1),ydata(plot_idx,2), 300, 'r', 'filled');
    
    figure
    set(gca,'FontSize',30)
    scatter(ydata(arange(),1),ydata(arange(),2),'MarkerEdgeColor',concat([0,0.5,0.5]),'MarkerFaceColor',concat([0,0.7,0.7]),'LineWidth',1.5)
    plot_names=2
# main.m:889
    # Overlay names
    if plot_names == 1:
        dx=0.75
# main.m:893
        dy=0.1
# main.m:893
        t=text(ydata(plot_idx,1) + dx,ydata(plot_idx,2) + dy,names_train(subidx(plot_idx)))
# main.m:894
        set(t,'FontSize',30)
        set(t,'FontWeight','bold')
    else:
        dx=0.1
# main.m:898
        dy=0.1
# main.m:898
        t=text(ydata(arange(),1) + dx,ydata(arange(),2) + dy,names_train(subidx))
# main.m:899
        set(t,'FontSize',30)
    
    xlabel('PC1')
    ylabel('PC2')
    title('t-SNE projection sources latent space P')
    hold('off')