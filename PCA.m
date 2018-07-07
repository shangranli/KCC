function [p_d_mat,project_mat,allmean]=PCA(initialize_mat,p)

%输入:
     %  initialize_mat: 原高维矩阵
     %  p: 要求降维后矩阵的维数
%输出:
     %p_d_mat: 降维后的p维矩阵
     %project_mat: 投影矩阵
     %allmean: 均值
%%%%%%%%%%%%%%%%%%%%%  lisr666@qq.com  %%%%%%%%%%%%%%%%%%%%%%%%%   
%%%%%%%%%%%%%%%%%%%%%%  2017/5/16  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    [features,rows]=size(initialize_mat);%原矩阵的特征(维)数，和列数
    allmean=mean(initialize_mat,2);%按列求原矩阵的均值
    meanmat=zeros(features,rows);%初始化均值矩阵
    
    %计算差值，原矩阵所有列减去均值
    for i=1:rows  
        meanmat(:,i)=initialize_mat(:,i)-allmean;
    end
    
    %利用差值矩阵构建协方差矩阵，即C=meanmat*meanmat';
    %然后计算C的特征值和特征向量。由于C的维数过大，计算代价太大，
    %利用奇异值分解原理求得协方差矩阵C=meanmat*meanmat'的特征值和特征向量
    %即，利用求meanmat'*meanmat的特征值、向量求原协方差矩阵meanmat*meanmat'
    %的特征值和特征向量
    %[U,S,V]=svd(meanmat)
    
    [v,smat]=eig(meanmat'*meanmat);%求得V矩阵和特征值
    %把特征值取出，放到数组中，按升序排列
    w=1:rows;
    for i=1:rows
        w(i)=smat(i,i);
    end
    
    project_mat=zeros(features,p);%初始化投影矩阵
    max_v=v(:,rows-p+1:rows); %取最大的特征值对应的特征向量
    %计算奇异值
    sigma=sqrt(w(rows-p+1:rows));
    %计算投影矩阵，U_i=A*V_i/sqrt(sigma)
    for i=1:p
        project_mat(:,i)=meanmat*max_v(:,i)/sigma(i);
    end
    %计算降维后的矩阵
    p_d_mat=project_mat'*meanmat;
    %L_2 归一
    for i=1:size(p_d_mat,2)
        p_d_mat(:,i)=p_d_mat(:,i)/norm(p_d_mat(:,i));
    end
end