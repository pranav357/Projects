
%%%%% ���string ��id
%%%% rankΪһ��string Ԫ������
function id = find_string(rank, s)  
     
     id = -1;
     for i = 1:length(rank)
         if isequal( rank{i}, s)
         id = i;
         end
     end
 
end

