
%%%%%%%%%%%%%% ͳ�ƶȺ����� rank ֵ�Ĺ�ϵ %%%%%%%%%%%%%%
%%%% rank����Ϊpreciision �� rating ��  %%%%%%%%%%%%%% 
%%%%%%%%%%%% ���룺�����û��Ķ�+�����û���rank  ������ %%%%%%%%%
%%%%%%%%%%%%% ���������ȵ��û�ȡ��rank��ƽ��ֵ  Ȼ������������Լ����Ӧ��ƽ����rankֵ%%%%%

%%%%%%%%%%% ע���ʱҪͳ�Ƹ��Ե���� �Ա���Ƴ� errorbar  belowΪͼ���·��ľ��� Ҳ������Сֵ��ƽ��ֵ��ƫ�� above��Ϊ���ֵ�����ƫ�� %%%%%%%%
%%%%  middle Ϊ��λ�� %%%%%%
function [x, average,middle] = distribution1(degree,rank)
  
	avi1 = degree;
	avi3 = rank; %%%%ÿ���û�������ָ��ֵ
	avi3(isnan(avi3)) = 0;
	avi3(isinf(avi3)) = 0;
	avi4 = [avi1,avi3]; %%2��
	x = unique(avi1); %%%�ȵķֲ�
	average = zeros(size(x)); 
    middle = [];
	for i = 1:length(x)
	    count = 0;
	    for j = 1:length(avi4)
            if avi4(j, 1) == x(i, 1);
               count = count + 1;
               average(i, 1) = average(i,1) + avi4(j,2);
            end
	    end
	    average(i, 1) = average(i, 1)/count;
	    clear count;
        sat = find( degree== x(i) );
        middle = [middle; median( rank(sat) )];
 
	end
	average(isnan( average )) = 0;
	average(isinf( average )) = 0;
end