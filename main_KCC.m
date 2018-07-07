tic
clear;
%% 
%%%%%%%%%%%%%%%%% orl人脸库 %%%%%%%%%%%%%%%%%
%下载数据集
disp('****下载数据集...****');
download_ORL();

load ORL_dat1;
p=[10,25,50,80]; 

allfor=20;

KCC_Gkernel_r=zeros(1,4);
KCC_Pkernel_r=zeros(1,4);
for j = 1:allfor
    disp(j);
randnum=randperm(10);
train_rand=[];
test_rand=[];
tr_lab=[];
tt_lab=[];
for i=1:max(img_lab1)
    class_index=find(img_lab1==i);
    tr_rand_index=randnum(1:5);
    te_rand_index=randnum(6:10);
    tr_rand=class_index(tr_rand_index);
    te_rand=class_index(te_rand_index);
    train_rand=[train_rand tr_rand];
    test_rand=[test_rand te_rand];
    
    train_lab=img_lab1(class_index(1:5));
    test_lab=img_lab1(class_index(6:10));
    tr_lab=[tr_lab train_lab];
    tt_lab=[tt_lab test_lab];
end
tr_dat=img_dat1(:,train_rand);
tt_dat=img_dat1(:,test_rand);

[orl_KCC_Gkernel_r]=ORL_KCC_Gkernel(p,tr_dat,tt_dat,tr_lab,tt_lab);
[orl_KCC_Pkernel_r]=ORL_KCC_Pkernel(p,tr_dat,tt_dat,tr_lab);

KCC_Gkernel_r=KCC_Gkernel_r+orl_KCC_Gkernel_r;
KCC_Pkernel_r=KCC_Pkernel_r+orl_KCC_Pkernel_r;
end

KCC_Gkernel_r=KCC_Gkernel_r./allfor;
KCC_Pkernel_r=KCC_Pkernel_r./allfor;

plot(p,KCC_Gkernel_r,'k-d',p,KCC_Pkernel_r,'k-.d');
xlabel('feature dimension d');
ylabel('Recognition Rate');

toc
