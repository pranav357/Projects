

%%%%% ͳ��rank ֵ���ظ�ֵ�� �����ֲ�  
%%%% �������Ϊ������ %%%%%%%%
%%%%�����rank���� ֵ��������

function [result, count] = statistics(rank)  %%rank Ϊ������

    clear data result count result sat;
    data = rank; %%%%
    result = unique(data); %%%
    count = zeros(1, length(result));
    for i = 1:length(result)
        count(1,i) = length(find(rank==result(1,i))) / length(rank);     
    end
    count(isinf(count)) = 0;
    count(isnan(count)) = 0;

   [result, sat] = sort(result,2);%%��rankֵ��������
   count = count(sat);
%  �ۼƸ���
   for j = 1:length(count)
       count(1,j) = sum(count(1,j:length(count)));
   end
%     
end


