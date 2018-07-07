function [rerate_r]=ORL_KCC_Pkernel(p,tr_dat,tt_dat,tr_lab)
kappa=0.05;
gamma=0.01;
k=1;
d=2;
rerate_r=p;
rate=1;
for f=p
    right_r=0;
    
    %训练集pca降维
    [Dictionary,project_mat,allmean]=PCA(tr_dat,f); 
    
    [B_T_B]=Polynomial_kernel(Dictionary,Dictionary,d);
  
    r=1:max(tr_lab);%存放残差
    %% 
    for i=1:size(tt_dat,2)%开始测试
        if ~mod(i,100)
            disp(['test,no.',num2str(i),',all,',num2str(size(tt_dat,2))]);
        end
        testvector=tt_dat(:,i)-allmean;
        protest=project_mat'*testvector;
        protest=protest/norm(protest);
        
        knnidx = knnsearch(Dictionary',protest','K',k, 'Distance', 'euclidean');
        for m = 1:k
            classidx=ceil(knnidx(:,m)/5);
            if m == 1
                Neigh_Coef = sum(Dictionary(:,classidx*5-4:classidx*5),2);
            else
                Neigh_Coef = Neigh_Coef + sum(Dictionary(:,classidx*5-4:classidx*5),2);
            end
        end
        Neigh_Coef = Neigh_Coef./k;
        Neigh_Coef=Neigh_Coef/norm(Neigh_Coef);
        fun=((1-gamma).*protest + gamma.*Neigh_Coef);
        
        [B_T_Phi]=Polynomial_kernel(Dictionary,fun,d);
       
        xp=(B_T_B+kappa*eye(size(B_T_B,2)))\B_T_Phi;
%% 
 %%%%%%%%%%%%%%%%%%%%%%%%%分类%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        for kk=1:max(tr_lab) 
            r(kk)=norm(B_T_Phi-B_T_B(:,tr_lab==kk)*xp(tr_lab==kk,:),2)^2/sum(xp(tr_lab==kk,:).*xp(tr_lab==kk,:));%计算残差
        end
        [min_r,index_r]=min(r);
        if index_r == tr_lab(i)
            right_r=right_r+1;
        end
          
    end  
    %% 
%%%%%%%%%%%%%%%%%%%%%% 计算识别率 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rerate_r(rate)=right_r/size(tt_dat,2);
    rate=rate+1;
end