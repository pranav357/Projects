
%%%%% ���������м���kendall tau�Ľ����ֵ
%%%���з�����ͬ������һ�� 
%%%%
function tau = cos_cal(data1, data2)  %% ������

    tau = sum(data1.*data2);
    tau = tau/sqrt(sum(data1.^2)*sum(data2.^2));

end
