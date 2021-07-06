def solution(l):
    nos_list = l
    nos_list.sort(reverse=True)

    def find_no(nos_list, rem):
        nos_list.sort()
        for nos in nos_list:
            if nos%3 == rem:
                nos_list.remove(nos)
                break
        if sum(nos_list)%3 == 0:
            nos_list.sort(reverse = True)
            fin_no = int(''.join(str(ints) for ints in nos_list))
        else:    
            rem_pair = []
            for i in range(0, len(nos_list)):
                    for j in range(i+1, len(nos_list)):
                            if (nos_list[i] + nos_list[j] )%3 == rem:
                                rem_pair.append(nos_list[i])
                                rem_pair.append(nos_list[j])
                                nos_list = [num for num in nos_list if num not in rem_pair]
                                break
            if sum(nos_list)%3 == 0:
                nos_list.sort(reverse = True)
                fin_no = int(''.join(str(ints) for ints in nos_list))
            elif any(nos%3 == 0 for nos in nos_list):
                nos_list = [num for num in nos_list if num%3 == 0]
                nos_list.sort(reverse = True)
                fin_no = int(''.join(str(ints) for ints in nos_list))
            else:
                fin_no = 0    
        return fin_no

    if sum(nos_list) == 0 or (len(nos_list) == 1 and sum(nos_list)%3 != 0):
        fin_no = 0  
    elif sum(nos_list)%3 == 0:
        fin_no = int(''.join(str(ints) for ints in nos_list))
    elif sum(nos_list)%3 != 0:
        rem = sum(nos_list)%3
        fin_no = find_no(nos_list, rem)

    return fin_no

l = [5,3,8,6]
sum(l)

solution(l)


        for rem_nos in [1,4,7]:
            if rem_nos in nos_unq:
                nos_unq.remove(rem_nos)
                break
            else:

            else: 
                fin_no = 0
        fin_no = int(''.join(str(ints) for ints in nos_unq))        
    elif sum(nos_unq)%3 == 2:
        for rem_nos in [2,5,8]:
            if rem_nos in nos_unq:
                nos_unq.remove(rem_nos)
                break
            else: 
                fin_no = 0
        fin_no = int(''.join(str(ints) for ints in nos_unq))     


setA = [161, 182, 161, 154, 176, 170, 167, 171, 170, 174]

savg = sum(set(setA))/len(set(setA))

