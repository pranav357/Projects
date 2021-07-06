
% 
% ������ϵ ���ϵı��� ������һ����һ�ģ� 
% ���в�ϵ  �������ϵ�ʹ�ô���
% recipe id ��0��ʼ  ���ϵ�id��1��ʼ
caixi_name = {'lucai','chuancai','yuecai','sucai','mincai','zhecai','xiangcai','huicai','dongbeicai','gangtai','hubeicai','hucai','jiangxicai','jingcai','other',...
	    		  'qingzhencai','shanxicai','xibeicai','yucai','yunguicai'};
num_recipe =  [1066,1148,775,372,468,460,691,218,358,151,160,744,143,606,52,521,125,188,173,79];
num_component = 2911;              
frequency_recipe =[];
frequency_recipe_guiyi = [];
frequency_ing = zeros(1, num_component);
N_ing = [];
Avg_ing = [];
Avg_ing_all = 0;
for i = 1:length(caixi_name)
     clear data_network data_recipe_id network temp; 
     data_network = load( strcat('data/network/network/',caixi_name{i},'_network.txt'));
     network = zeros( num_recipe(i), num_component);
     
     for k = 1:length(data_network)
        network( data_network(k, 1)+1, data_network(k, 2)) = 1;    
     end 
     Avg_ing_all = Avg_ing_all + nnz(network);
     temp = sum(network, 1);
     N_ing = [N_ing, nnz(temp)];
     Avg_ing = [Avg_ing, mean(sum(network,2))];
     frequency_recipe = [frequency_recipe; temp./num_recipe(i)];
     frequency_recipe_guiyi = [ frequency_recipe_guiyi; temp/nnz(network)];
     frequency_ing = frequency_ing + temp;
end
disp('����ϵ�����ϲ�ͬ������'); N_ing
disp('����ϵÿ�����������ϵ�ƽ��ֵ'); Avg_ing
disp('ȫ��������recipeƽ�����ϸ�����'); Avg_ing_all/sum(num_recipe)
%%%%frequency_recipe 2911*20���� 
dlmwrite( strcat('data/real_result/', 'frequency_component_guiyi.txt'),frequency_recipe_guiyi',' '); %%%��һ���� 
dlmwrite( strcat('data/real_result/', 'frequency_component.txt'),frequency_recipe',' '); %%%δ��һ���� 
dlmwrite( strcat('data/real_result/', 'frequency_ingredient_all.txt'),frequency_ing',' ');
 