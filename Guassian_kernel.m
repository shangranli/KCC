function [k_matrix]=Guassian_kernel(matrix1,matrix2,t)
    k_matrix=zeros(size(matrix1,2),size(matrix2,2));
    
    for i=1:size(matrix1,2)
        for j=1:size(matrix2,2)
            k_matrix(i,j)=exp(-((matrix1(:,i)-matrix2(:,j))'*(matrix1(:,i)-matrix2(:,j)))/t);
        end
    end
end
