function [k_matrix]=Polynomial_kernel(matrix1,matrix2,d)
    k_matrix=zeros(size(matrix1,2),size(matrix2,2));
    
    for i=1:size(matrix1,2)
        for j=1:size(matrix2,2)
            k_matrix(i,j)=(1+matrix1(:,i)'*matrix2(:,j))^d;
        end
    end
end