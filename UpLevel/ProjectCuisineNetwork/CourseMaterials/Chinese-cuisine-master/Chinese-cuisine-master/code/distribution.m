
%%%%%%%%%%%%%% ͳ�ƶȺ����� rank ֵ�Ĺ�ϵ %%%%%%%%%%%%%%
%%%% rank����Ϊpreciision �� rating ��  %%%%%%%%%%%%%% 
%%%%%%%%%%%% ���룺�����û��Ķ�+�����û���rank  ������ %%%%%%%%%
%%%%%%%%%%%%% ���������ȵ��û�ȡ��rank��ƽ��ֵ  Ȼ������������Լ����Ӧ��ƽ����rankֵ%%%%%

%%%%%%%%%%% ע���ʱҪͳ�Ƹ��Ե���� �Ա���Ƴ� errorbar  belowΪͼ���·��ľ��� Ҳ������Сֵ��ƽ��ֵ��ƫ�� above��Ϊ���ֵ�����ƫ�� %%%%%%%%

function [x, average, below, above, error] = distribution(degree,rank)
  
	avi1 = degree;
	avi3 = rank; %%%%ÿ���û�������ָ��ֵ
	avi3(isnan(avi3)) = 0;
	avi3(isinf(avi3)) = 0;
	avi4 = [avi1,avi3]; %%2��
	x = unique(avi1); %%%�ȵķֲ�
	average = zeros(size(x)); 
    below = max(max(rank)) * ones(size(x));
    above = min(min(rank)) * ones(size(x)); 
    error = [];
	for i = 1:length(x)
	    count = 0;
	    for j = 1:length(avi4)
            if avi4(j, 1) == x(i, 1);
               count = count + 1;
               average(i, 1) = average(i,1) + avi4(j,2);
               below(i, 1) = min(below(i,1), avi4(j,2)); %%����Сֵ		  
               above(i, 1) = max(above(i,1), avi4(j,2));	%%%�������ֵ
            end
	    end
	    average(i, 1) = average(i, 1)/count;
        %%%%���㷽��
        clear sat
        sat = find(degree == x(i,1));
        sat = rank(sat);
        sat = sat  - average(i, 1) ;
        sat = sat.^2;
        sat = mean(sat);
        error = [error; sat];
        %%%%%%%
        
	    clear count;
	end
	average(isnan( average )) = 0;
	average(isinf( average )) = 0;

    below = average - below ;%%%��ͼ���·��ľ���
	above = above - average; %%��ͼ���Ϸ��ľ��� 

end