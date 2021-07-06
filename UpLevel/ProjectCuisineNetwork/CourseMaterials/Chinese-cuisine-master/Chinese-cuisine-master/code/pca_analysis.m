

%%%%%%-----PCA

data = load('data/real_result/frequency_component_guiyi.txt'); %%% 2911�� 20��
data = data'; %%% 20�� 2911�� 
[cof,score,latent,t] = princomp(data);
latent = cumsum(latent)./sum(latent); %%%  ����ÿ�����ݵ��ۻ�����ֵ  ȡʹ�ù����ʴ���95%�ĵ�һ��ֵ���� 
pca_num = min( find(latent>0.95));
%pca_num = 2;
tran = cof(:,1:pca_num); %%% cof��ǰpca_num�м�Ϊ ת������

pca_data = data * tran; %%����*ת������ = ԭ���������ɷֿռ�����ķֲ�
dlmwrite('pca.txt',pca_data,' ');  %%% 20�� 13�� 

% %3D 
% surfc(pca_data(1:20,1:3));
% colormap hsv;
% %% 

for i= 1:20
   scatter(pca_data(i,1),pca_data(i,2))
   hold on
end

% for i = 1:size(pca_data,1)
% plot(1:size(pca_data,2) ,pca_data(i,:));
% hold on;
% end